from functools import partial
from typing import Optional, Sequence

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from ml_collections import config_dict

from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    DDPM,
    MLP,
    DiffusionMLP,
    DiffusionMLPResNet,
    Ensemble,
    FourierFeatures,
    MLPResNetV2,
    StateActionValue,
    cosine_beta_schedule,
    ddim_sampler,
    subsample_ensemble,
    vp_beta_schedule,
)


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=("critic_fn",))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    return q_values.min(axis=0)


@partial(jax.jit, static_argnames=("apply_fn",))
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@jax.jit
def _sample_actions_jit(agent, observations):
    rng = agent.rng
    observations = jnp.squeeze(observations)
    assert observations.ndim == 1, observations.shape
    observations = jnp.expand_dims(observations, axis=0)

    actor_params = agent.target_actor.params
    if agent.filter_enabled and agent.filter_at_eval:
        actions, _, rng = agent._sample_candidates(rng, observations, agent.N, actor_params, agent.filter_temperature_train)
    else:
        observations_repeated = jnp.broadcast_to(
            observations[:, None, :], (observations.shape[0], agent.N, observations.shape[-1])
        ).reshape(-1, observations.shape[-1])
        actions, rng = ddim_sampler(
            agent.actor.apply_fn,
            actor_params,
            agent.T,
            rng,
            agent.action_dim,
            observations_repeated,
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.M,
            eta=agent.ddim_eta,
        )
        actions = actions.reshape(1, agent.N, -1)

    diffusion_actions = actions.squeeze(axis=0)

    if diffusion_actions.shape[0] > 1 or agent.ne_samples > 0:
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, agent.target_critic.params, agent.num_min_qs, agent.num_qs)
        obs_rep = jnp.broadcast_to(
            observations[:, None, :], (observations.shape[0], diffusion_actions.shape[0], observations.shape[-1])
        ).reshape(-1, observations.shape[-1])
        all_actions = diffusion_actions

        if agent.ne_samples > 0:
            key, rng = jax.random.split(rng)
            d_actions = diffusion_actions[: agent.ne_samples]
            r_obs = jnp.broadcast_to(observations[:, None, :], (observations.shape[0], agent.ne_samples, observations.shape[-1])).reshape(
                -1, observations.shape[-1]
            )
            r_in = jnp.concatenate([r_obs, d_actions], axis=1)
            r_samples, rng = _sample_actions(key, agent.edit_actor.apply_fn, agent.edit_actor.params, r_in)
            r_samples = agent._apply_residual_action_mask(r_samples * agent.r_action_scale) + d_actions
            r_samples = jnp.clip(r_samples, -1.0, 1.0)
            obs_rep = jnp.concatenate([obs_rep, r_obs], axis=0)
            all_actions = jnp.concatenate([all_actions, r_samples], axis=0)

        qs = compute_q(agent.target_critic.apply_fn, target_params, obs_rep, all_actions)
        idx = jnp.argmax(qs)
        action = all_actions[idx]
    else:
        action = diffusion_actions[0]

    rng, _ = jax.random.split(rng)
    return action, rng


_ALLOWED_SAMPLING_MODES = ("zscore", "plain")


def _validate_sampling_mode(mode: str) -> None:
    if mode not in _ALLOWED_SAMPLING_MODES:
        raise ValueError(f"Invalid sampling_mode={mode}. Allowed: {_ALLOWED_SAMPLING_MODES}")


def _z_score_normalize(values: jnp.ndarray, axis: int, eps: float = 1e-6) -> jnp.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / jnp.maximum(std, eps)


def _gumbel_topk(key, logits, k):
    u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    _, idx = jax.lax.top_k(logits + g, k)
    return idx


def sample_k_indices(key, scores: jnp.ndarray, k: int, *, temperature: jnp.ndarray, mode: str = "zscore") -> jnp.ndarray:
    _validate_sampling_mode(mode)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got k={k}")

    scores = jnp.asarray(scores)
    if scores.ndim < 1:
        raise ValueError(f"scores must have ndim >= 1; got shape={scores.shape}")

    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"k must be <= scores.shape[-1]; got k={k}, n={n}")

    prefix = scores.shape[:-1]
    batch = int(np.prod(prefix)) if prefix else 1
    s2 = scores.reshape(batch, n)

    proc = s2 if mode == "plain" else _z_score_normalize(s2, axis=1)

    temp = jnp.asarray(temperature, dtype=proc.dtype)
    temp = jnp.broadcast_to(temp, (batch,))
    do_sample = temp > 0
    temp_safe = jnp.where(do_sample, temp, jnp.asarray(1.0, dtype=proc.dtype))
    logits = proc / temp_safe[:, None]

    idx_sample = _gumbel_topk(key, logits, k)
    idx_det = jax.lax.top_k(s2, k)[1]
    idx = jnp.where(do_sample[:, None], idx_sample, idx_det)
    return idx.reshape(prefix + (k,))


def _gather_axis1(x, idx):
    if x is None:
        return None
    idx = jnp.asarray(idx)
    batch = jnp.arange(x.shape[0]).reshape((x.shape[0],) + (1,) * (idx.ndim - 1))
    batch = jnp.broadcast_to(batch, idx.shape)
    return x[batch, idx]


default_init = nn.initializers.xavier_uniform


class DenoisingStateActionValue(nn.Module):
    base_cls: nn.Module
    cond_encoder_cls: nn.Module
    time_preprocess_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, time: jnp.ndarray, training: bool = False):
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        inputs = jnp.concatenate([observations, actions, cond], axis=-1)
        outputs = self.base_cls()(inputs, training=training)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


@partial(jax.jit, static_argnames=("actor_apply_fn", "hidden_apply_fn", "act_dim", "T", "N", "training", "filter_temperature_mode"))
def ddim_sampler_hidden_filter(
    actor_apply_fn,
    actor_params,
    hidden_apply_fn,
    hidden_params,
    T,
    N,
    rng,
    act_dim,
    observations,
    alphas,
    alpha_hats,
    betas,
    repeat_last_step,
    filter_temperature: float = 0.0,
    filter_temperature_mode: str = "plain",
    ddim_eta: float = 0.0,
    training: bool = False,
    init_noise=None,
):
    batch_size = observations.shape[0]
    obs = jnp.broadcast_to(observations[:, None, :], (batch_size, N, observations.shape[-1]))
    rng, init_key, noise_key = jax.random.split(rng, 3)
    x = jax.random.normal(init_key, (batch_size, N, act_dim)) if init_noise is None else init_noise
    temp = jnp.asarray(filter_temperature, dtype=jnp.float32)

    def step(x_, obs_, t_):
        x_flat = x_.reshape(-1, act_dim)
        obs_flat = obs_.reshape(-1, obs_.shape[-1])
        t_ = jnp.asarray(t_, dtype=jnp.int32)
        time = jnp.full((x_flat.shape[0], 1), t_)
        eps_pred = actor_apply_fn({"params": actor_params}, obs_flat, x_flat, time, training=training)

        def ddim_step(_):
            alpha_hat_t = alpha_hats[t_]
            sqrt_alpha_hat_t = jnp.sqrt(alpha_hat_t)
            sqrt_one_minus_alpha_hat_t = jnp.sqrt(1.0 - alpha_hat_t)
            x0_pred = (x_flat - sqrt_one_minus_alpha_hat_t * eps_pred) / sqrt_alpha_hat_t
            alpha_hat_prev = jax.lax.cond(t_ > 0, lambda t: alpha_hats[t - 1], lambda _: jnp.asarray(1.0, dtype=alpha_hat_t.dtype), t_)
            eta = jnp.asarray(ddim_eta, dtype=x_flat.dtype)

            def deterministic(_):
                x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + jnp.sqrt(1.0 - alpha_hat_prev) * eps_pred
                return x_next, x0_pred

            def stochastic(_):
                sigma = eta * jnp.sqrt((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev))
                z = jax.random.normal(jax.random.fold_in(noise_key, t_), x_flat.shape)
                noise = sigma * z
                eps_scale = jnp.sqrt(jnp.maximum(0.0, 1.0 - alpha_hat_prev - sigma**2))
                x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + eps_scale * eps_pred + noise
                return x_next, x0_pred

            return jax.lax.cond(eta == 0.0, deterministic, stochastic, operand=None)

        x_next, x_eval = ddim_step(None)
        x_next = jnp.clip(x_next, -1, 1)
        x_eval = jnp.clip(x_eval, -1, 1)
        return x_next.reshape(x_.shape), x_eval.reshape(x_.shape)

    def denoise_segment(x_, obs_, t_start, t_stop_exclusive):
        times = jnp.arange(t_start, t_stop_exclusive, -1)

        def body(x__, t_):
            x__, _ = step(x__, obs_, t_)
            return x__, ()

        x_, _ = jax.lax.scan(body, x_, times)
        return x_

    x_eval = x
    stored = (x_eval,)
    if x.shape[1] > 1:
        if hidden_apply_fn is None or hidden_params is None:
            raise ValueError("hidden_apply_fn and hidden_params are required when filtering more than one candidate.")
        x_flat = x_eval.reshape(-1, act_dim)
        obs_flat = obs.reshape(-1, obs.shape[-1])
        time = jnp.full((x_flat.shape[0], 1), T, dtype=jnp.int32)
        q_values_sel = hidden_apply_fn({"params": hidden_params}, obs_flat, x_flat, time)
        q_sel = q_values_sel.min(axis=0).reshape(batch_size, -1)
        rng, key = jax.random.split(rng)

        def select_idx(key_, scores_, k_):
            return sample_k_indices(key_, scores_, k_, temperature=temp, mode=filter_temperature_mode)

        k_keep = 1
        idx = select_idx(key, q_sel, k_keep)
        x = _gather_axis1(x, idx)
        obs = _gather_axis1(obs, idx)
        stored = tuple(_gather_axis1(s, idx) for s in stored)

    if T > 0:
        x = denoise_segment(x, obs, T - 1, -1)

    def repeat_body(_, x_):
        x_, _ = step(x_, obs, 0)
        return x_

    x = jax.lax.fori_loop(0, repeat_last_step, repeat_body, x)

    x = jnp.clip(x, -1, 1)
    return x, stored, rng


class BetterDiffusionSACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    target_actor: TrainState
    edit_actor: TrainState
    temp: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    filter_critic: Optional[TrainState]
    target_filter_critic: Optional[TrainState]
    filter_enabled: bool = struct.field(pytree_node=False)
    filter_at_eval: bool = struct.field(pytree_node=False)
    filter_temperature_train: float = struct.field(pytree_node=False)
    filter_temperature_eval: float = struct.field(pytree_node=False)
    filter_temperature_mode: str = struct.field(pytree_node=False)

    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    train_N: int = struct.field(pytree_node=False)
    ne_samples: int = struct.field(pytree_node=False)
    ne_samples_train: int = struct.field(pytree_node=False)
    r_action_scale: float = struct.field(pytree_node=False)
    residual_action_mask: Optional[np.ndarray] = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddim_eta: float = struct.field(pytree_node=False)
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float

    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    filter_num_min_qs: Optional[int] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Optional[Sequence[int]] = None,
        outer_critic_hidden_dims: Optional[Sequence[int]] = None,
        filter_critic_hidden_dims: Optional[Sequence[int]] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        filter_num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        init_temperature: float = 1.0,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None,
        d_actor_drop: Optional[float] = None,
        T: int = 10,
        N: int = 32,
        train_N: int = 32,
        M: int = 0,
        ne_samples: int = 0,
        ne_samples_train: int = 0,
        r_action_scale: float = 1.0,
        residual_action_mask: Optional[Sequence[float]] = None,
        actor_layer_norm: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.001,
        actor_num_blocks: int = 3,
        ddim_eta: float = 0.0,
        beta_schedule: str = "vp",
        filter_enabled: bool = False,
        filter_at_eval: bool = False,
        filter_temperature_train: Optional[float] = None,
        filter_temperature_eval: Optional[float] = None,
        filter_temperature_mode: str = "plain",
    ):
        action_dim = action_space.shape[-1]
        if residual_action_mask is not None:
            residual_action_mask = np.asarray(residual_action_mask, dtype=np.float32)
            if residual_action_mask.shape != (action_dim,):
                raise ValueError(f"Expected residual_action_mask shape ({action_dim},), got {residual_action_mask.shape}")

        ddim_eta = float(ddim_eta)
        assert ddim_eta >= 0.0

        target_num_qs = num_min_qs if num_min_qs is not None else num_qs
        target_filter_num_qs = filter_num_min_qs if filter_num_min_qs is not None else num_qs

        assert target_num_qs >= 2, target_num_qs
        _validate_sampling_mode(filter_temperature_mode)

        if isinstance(action_space, gym.Space):
            observations = observation_space.sample()
            actions = action_space.sample()
        else:
            observations = observation_space
            actions = action_space
        hidden_dims = tuple(hidden_dims)
        critic_hidden_dims = hidden_dims if critic_hidden_dims is None else tuple(critic_hidden_dims)
        outer_critic_hidden_dims = critic_hidden_dims if outer_critic_hidden_dims is None else tuple(outer_critic_hidden_dims)
        filter_critic_hidden_dims = critic_hidden_dims if filter_critic_hidden_dims is None else tuple(filter_critic_hidden_dims)
        if target_entropy is None:
            if adjust_target_entropy:
                target_entropy = -action_dim / 2 + action_dim * jnp.log(r_action_scale)
            else:
                target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(DiffusionMLP, hidden_dims=(time_dim * 2, time_dim * 2), activations=nn.swish, activate_final=False)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        base_model_cls = partial(
            DiffusionMLPResNet,
            use_layer_norm=actor_layer_norm,
            num_blocks=actor_num_blocks,
            dropout_rate=d_actor_drop,
            out_dim=action_dim,
            activations=nn.swish,
        )
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls, cond_encoder_cls=cond_model_cls, reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(jnp.asarray(observations), axis=0)
        actions = jnp.expand_dims(jnp.asarray(actions), axis=0)
        actor_params = actor_def.init(actor_key, observations, actions, time)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adamw(learning_rate=actor_lr))
        target_actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[: i + 1]) for i in range(T)])

        edit_actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm)
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate([observations, jnp.ones((1, action_dim))], axis=1)
        edit_actor_params = edit_actor_def.init(actor_key, edit_observations)["params"]
        edit_actor = TrainState.create(apply_fn=edit_actor_def.apply, params=edit_actor_params, tx=optax.adam(learning_rate=actor_lr))

        def make_critic_base_cls(critic_hidden_dims_):
            if use_critic_resnet:
                return partial(MLPResNetV2, num_blocks=1)
            return partial(
                MLP,
                hidden_dims=critic_hidden_dims_,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )

        def make_outer_critic_cls(critic_hidden_dims_):
            return partial(StateActionValue, base_cls=make_critic_base_cls(critic_hidden_dims_))

        def make_filter_critic_cls(critic_hidden_dims_):
            return partial(
                DenoisingStateActionValue,
                base_cls=make_critic_base_cls(critic_hidden_dims_),
                cond_encoder_cls=cond_model_cls,
                time_preprocess_cls=preprocess_time_cls,
            )

        outer_critic_cls = make_outer_critic_cls(outer_critic_hidden_dims)
        filter_critic_cls = make_filter_critic_cls(filter_critic_hidden_dims)
        critic_def = Ensemble(outer_critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay, mask=decay_mask_fn)
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=tx)
        target_critic_def = Ensemble(outer_critic_cls, num=target_num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr))

        filter_critic = None
        target_filter_critic = None
        htemp_train = 0.0
        htemp_eval = 0.0

        if filter_enabled:
            assert train_N >= 1 and ne_samples_train <= 1
            if filter_at_eval:
                assert N >= 1 and ne_samples <= 1
            htemp_train = 1.0 if filter_temperature_train is None else float(filter_temperature_train)
            htemp_eval = 1.0 if filter_temperature_eval is None else float(filter_temperature_eval)

            rng, hidden_key = jax.random.split(rng)
            hidden_def = Ensemble(filter_critic_cls, num=num_qs)
            hidden_params = hidden_def.init(hidden_key, observations, actions, time)["params"]
            filter_critic = TrainState.create(apply_fn=hidden_def.apply, params=hidden_params, tx=optax.adam(learning_rate=critic_lr))
            target_hidden_def = Ensemble(filter_critic_cls, num=target_filter_num_qs)
            target_filter_critic = TrainState.create(
                apply_fn=target_hidden_def.apply, params=hidden_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
            )

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            edit_actor=edit_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            action_dim=action_dim,
            T=T,
            N=N,
            train_N=train_N,
            ne_samples=ne_samples,
            ne_samples_train=ne_samples_train,
            r_action_scale=r_action_scale,
            residual_action_mask=residual_action_mask,
            M=M,
            actor_tau=actor_tau,
            ddim_eta=float(ddim_eta),
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            filter_num_min_qs=filter_num_min_qs,
            filter_critic=filter_critic,
            target_filter_critic=target_filter_critic,
            filter_enabled=filter_enabled,
            filter_at_eval=filter_at_eval,
            filter_temperature_train=htemp_train,
            filter_temperature_eval=htemp_eval,
            filter_temperature_mode=filter_temperature_mode,
        )

    def _sample_candidates(self, rng, observations, N, actor_params, filter_temperature):
        observations = jax.device_put(jnp.asarray(observations))
        if observations.ndim == 1:
            observations = observations[None, :]
        batch_size = observations.shape[0]

        if not self.filter_enabled:
            observations_repeated = jnp.broadcast_to(observations[:, None, :], (batch_size, N, observations.shape[-1])).reshape(
                -1, observations.shape[-1]
            )
            actions_flat, rng = ddim_sampler(
                self.actor.apply_fn,
                actor_params,
                self.T,
                rng,
                self.action_dim,
                observations_repeated,
                self.alphas,
                self.alpha_hats,
                self.betas,
                self.M,
                eta=self.ddim_eta,
            )
            actions = actions_flat.reshape(batch_size, N, -1)
            return actions, (), rng

        hidden_apply_fn = None
        hidden_params = None
        if N > 1:
            assert self.target_filter_critic is not None
            key, rng = jax.random.split(rng)
            hidden_apply_fn = self.target_filter_critic.apply_fn
            hidden_params = subsample_ensemble(key, self.target_filter_critic.params, self.filter_num_min_qs, self.num_qs)

        actions, stored, rng = ddim_sampler_hidden_filter(
            self.actor.apply_fn,
            actor_params,
            hidden_apply_fn,
            hidden_params,
            self.T,
            N,
            rng,
            self.action_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.M,
            filter_temperature=filter_temperature,
            filter_temperature_mode=self.filter_temperature_mode,
            ddim_eta=self.ddim_eta,
            init_noise=None,
        )
        return actions, stored, rng

    def _outer_backup_q_scores(self, target_params, observations, actions):
        q_values_sel = self.target_critic.apply_fn({"params": target_params}, observations, actions)
        q_sel_values = q_values_sel[0::2]
        q_eval_values = q_values_sel[1::2]
        if q_sel_values.shape[0] == 0:
            q_sel_values = q_values_sel
        if q_eval_values.shape[0] == 0:
            q_eval_values = q_values_sel

        q_sel = q_sel_values.min(axis=0)
        q_eval = q_eval_values.min(axis=0)
        return q_sel, q_eval

    def _apply_residual_action_mask(self, residual_actions):
        if self.residual_action_mask is None:
            return residual_actions
        mask = jnp.asarray(self.residual_action_mask, dtype=residual_actions.dtype)
        return residual_actions * mask

    def _select_best_actions(self, rng, observations, actions, target_params):
        batch_size = observations.shape[0]
        num_candidates = actions.shape[1]
        total_candidates = num_candidates + self.ne_samples_train
        observations_repeated = jnp.broadcast_to(observations[:, None, :], (batch_size, total_candidates, observations.shape[-1])).reshape(
            -1, observations.shape[-1]
        )
        actions_all = actions

        if self.ne_samples_train > 0:
            key, rng = jax.random.split(rng)
            r_observations = jnp.broadcast_to(
                observations[:, None, :], (batch_size, self.ne_samples_train, observations.shape[-1])
            ).reshape(-1, observations.shape[-1])
            d_actions = actions[:, : self.ne_samples_train].reshape(-1, actions.shape[-1])
            r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
            r_samples, rng = _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
            r_samples = self._apply_residual_action_mask(r_samples * self.r_action_scale) + d_actions
            r_samples = jnp.clip(r_samples, -1.0, 1.0)
            r_samples = r_samples.reshape(batch_size, self.ne_samples_train, -1)
            actions_all = jnp.concatenate([actions_all, r_samples], axis=1)

        actions_flat = actions_all.reshape(-1, actions_all.shape[-1])
        obs_flat = observations_repeated

        q_sel, q_eval = self._outer_backup_q_scores(target_params, obs_flat, actions_flat)

        q_sel = q_sel.reshape(batch_size, num_candidates + self.ne_samples_train)
        q_eval = q_eval.reshape(batch_size, num_candidates + self.ne_samples_train)

        best_indices = jnp.argmax(q_sel, axis=1)

        batch_indices = jnp.arange(batch_size)
        best_actions = actions_all[batch_indices, best_indices]
        best_q = q_eval[batch_indices, best_indices]
        return best_actions, best_q, best_indices, rng

    def eval_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0)

        actor_params = self.target_actor.params
        if self.filter_enabled and self.filter_at_eval:
            actions, _, rng = self._sample_candidates(rng, observations, self.N, actor_params, self.filter_temperature_eval)
        else:
            observations_repeated = jnp.broadcast_to(
                observations[:, None, :], (observations.shape[0], self.N, observations.shape[-1])
            ).reshape(-1, observations.shape[-1])
            actions, rng = ddim_sampler(
                self.actor.apply_fn,
                actor_params,
                self.T,
                rng,
                self.action_dim,
                observations_repeated,
                self.alphas,
                self.alpha_hats,
                self.betas,
                self.M,
                eta=self.ddim_eta,
            )
            actions = actions.reshape(1, self.N, -1)

        diffusion_actions = actions.squeeze(axis=0)

        if diffusion_actions.shape[0] > 1 or self.ne_samples > 0:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)
            obs_rep = jnp.broadcast_to(
                observations[:, None, :], (observations.shape[0], diffusion_actions.shape[0], observations.shape[-1])
            ).reshape(-1, observations.shape[-1])
            all_actions = diffusion_actions

            if self.ne_samples > 0:
                key, rng = jax.random.split(rng)
                d_actions = diffusion_actions[: self.ne_samples]
                r_obs = jnp.broadcast_to(
                    observations[:, None, :], (observations.shape[0], self.ne_samples, observations.shape[-1])
                ).reshape(-1, observations.shape[-1])
                r_in = jnp.concatenate([r_obs, d_actions], axis=1)
                r_samples, rng = _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_in)
                r_samples = self._apply_residual_action_mask(r_samples * self.r_action_scale) + d_actions
                r_samples = jnp.clip(r_samples, -1.0, 1.0)
                obs_rep = jnp.concatenate([obs_rep, r_obs], axis=0)
                all_actions = jnp.concatenate([all_actions, r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, obs_rep, all_actions)
            idx = jnp.argmax(qs)
            action = all_actions[idx]
        else:
            action = diffusion_actions[0]

        rng, _ = jax.random.split(rng)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def sample_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    @jax.jit
    def update_edit_actor(self, batch: DatasetDict):
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params):
            edit_observations = jnp.concatenate([batch["observations"], batch["actions"]], axis=1)
            dist = self.edit_actor.apply_fn({"params": actor_params}, edit_observations, training=True, rngs={"dropout": dropout_key})
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            actions = self._apply_residual_action_mask(actions * self.r_action_scale)
            log_probs = log_probs - actions.shape[-1] * jnp.log(self.r_action_scale)
            actions = actions + batch["actions"]
            actions = jnp.clip(actions, -1.0, 1.0)
            qs = self.critic.apply_fn({"params": self.critic.params}, batch["observations"], actions, True, rngs={"dropout": key2})
            q = qs.mean(axis=0)
            loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return loss, {"edit_q": q.mean(), "edit_actor_loss": loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)
        return self.replace(edit_actor=edit_actor, rng=rng), actor_info

    @jax.jit
    def update_actor(self, batch: DatasetDict):
        rng = self.rng
        key, rng = jax.random.split(rng)
        time = jax.random.randint(key, (batch["actions"].shape[0],), 0, self.T)
        key, rng = jax.random.split(rng)
        noise_sample = jax.random.normal(key, (batch["actions"].shape[0], self.action_dim))

        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng)

        def actor_loss_fn(score_model_params):
            eps_pred = self.actor.apply_fn(
                {"params": score_model_params}, batch["observations"], noisy_actions, time, rngs={"dropout": key}, training=True
            )
            loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return loss, {"actor_loss": loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        target_params = optax.incremental_update(actor.params, self.target_actor.params, self.actor_tau)
        target_actor = self.target_actor.replace(params=target_params)
        return self.replace(actor=actor, target_actor=target_actor, rng=rng), info

    @jax.jit
    def update_temperature(self, entropy: float):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    @jax.jit
    def update_critic_from_candidates(self, batch: DatasetDict, next_actions_candidates: jnp.ndarray, stored, rng):
        next_obs = batch["next_observations"]

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

        next_actions, next_q, _, rng = self._select_best_actions(rng, next_obs, next_actions_candidates, target_params)
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn({"params": critic_params}, batch["observations"], batch["actions"], True, rngs={"dropout": key})
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)

        filter_critic = self.filter_critic
        target_filter_critic = self.target_filter_critic
        if self.filter_enabled and stored and filter_critic is not None and target_filter_critic is not None:
            k_final = next_actions_candidates.shape[1]
            obs_rep = jnp.broadcast_to(next_obs[:, None, :], (next_obs.shape[0], k_final, next_obs.shape[-1])).reshape(
                -1, next_obs.shape[-1]
            )
            a0_flat = next_actions_candidates.reshape(-1, next_actions_candidates.shape[-1])
            q0 = compute_q(self.target_critic.apply_fn, target_params, obs_rep, a0_flat).reshape(-1, k_final)
            q0 = jax.lax.stop_gradient(q0)
            rng, drop_key = jax.random.split(rng)
            num_stages = len(stored)
            stored_actions = jnp.stack(stored, axis=0)
            a_stacked = stored_actions.reshape(-1, stored_actions.shape[-1])
            q0_flat = q0.reshape(-1)
            obs_stacked = jnp.broadcast_to(obs_rep[None, :, :], (num_stages, obs_rep.shape[0], obs_rep.shape[1])).reshape(
                -1, obs_rep.shape[-1]
            )
            t_stages = jnp.asarray((self.T,), dtype=jnp.int32)
            q_targets = jnp.broadcast_to(q0_flat[None, :], (num_stages, q0_flat.shape[0]))
            time_stacked = jnp.broadcast_to(t_stages[:, None, None], (num_stages, q0_flat.shape[0], 1)).reshape(-1, 1)

            def hidden_loss_fn(hidden_params):
                qs = filter_critic.apply_fn(
                    {"params": hidden_params}, obs_stacked, a_stacked, time_stacked, True, rngs={"dropout": drop_key}
                )
                actual_nq = qs.shape[0]
                qs_r = qs.reshape(actual_nq, num_stages, -1)
                diffs = qs_r - q_targets[None, :, :]
                stage_losses = (diffs**2).mean(axis=(0, 2))
                loss = stage_losses.mean()
                q_min = qs_r.min(axis=0)
                pred_mean = q_min.mean()
                q0_mean = q0.mean()
                return loss, {"filter_critic_loss": loss, "hidden_q": pred_mean, "hidden_target_q": q0_mean}

            grads, hidden_info = jax.grad(hidden_loss_fn, has_aux=True)(filter_critic.params)
            filter_critic = filter_critic.apply_gradients(grads=grads)
            target_hidden_params = optax.incremental_update(filter_critic.params, target_filter_critic.params, self.tau)
            target_filter_critic = target_filter_critic.replace(params=target_hidden_params)
            info.update(hidden_info)

        agent = self.replace(
            critic=critic, target_critic=target_critic, filter_critic=filter_critic, target_filter_critic=target_filter_critic, rng=rng
        )
        return agent, info

    @jax.jit
    def update_critic(self, batch: DatasetDict):
        rng = self.rng
        next_obs = batch["next_observations"]

        actor_params = self.actor.params
        next_actions_candidates, stored, rng = self._sample_candidates(
            rng, next_obs, self.train_N, actor_params, self.filter_temperature_train
        )
        return self.update_critic_from_candidates(batch, next_actions_candidates, stored, rng)

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_r"))
    def update_offline(self, batch: DatasetDict, utd_ratio: int, pretrain_q: bool, pretrain_r: bool):
        assert utd_ratio > 0
        assert batch["observations"].shape[0] % utd_ratio == 0
        mini_batch_size = batch["observations"].shape[0] // utd_ratio

        def get_mini_batch(i):
            start = i * mini_batch_size
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mini_batch_size, axis=0), batch)

        actor_batch = get_mini_batch(utd_ratio - 1)

        new_agent = self
        critic_info = {}
        if pretrain_q:

            def body(i, carry):
                agent, _ = carry
                mini_batch = get_mini_batch(i)
                agent, info = agent.update_critic(mini_batch)
                return agent, info

            mini_batch0 = get_mini_batch(0)
            new_agent, critic_info = new_agent.update_critic(mini_batch0)
            new_agent, critic_info = jax.lax.fori_loop(1, utd_ratio, body, (new_agent, critic_info))

        new_agent, actor_info = new_agent.update_actor(actor_batch)
        if pretrain_r and (self.ne_samples + self.ne_samples_train > 0):
            new_agent, edit_info = new_agent.update_edit_actor(actor_batch)
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            actor_info.update(edit_info)
            actor_info.update(temp_info)
        filter_noise_info = {}
        return new_agent, {**actor_info, **critic_info, **filter_noise_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_separate(self, batch: DatasetDict, actor_batch: DatasetDict, utd_ratio: int):
        assert utd_ratio > 0
        assert batch["observations"].shape[0] % utd_ratio == 0
        mini_batch_size = batch["observations"].shape[0] // utd_ratio

        def get_mini_batch(i):
            start = i * mini_batch_size
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mini_batch_size, axis=0), batch)

        mini_batch_last = get_mini_batch(utd_ratio - 1)

        def body(i, carry):
            agent, _ = carry
            mini_batch = get_mini_batch(i)
            agent, info = agent.update_critic(mini_batch)
            return agent, info

        mini_batch0 = get_mini_batch(0)
        new_agent, critic_info = self.update_critic(mini_batch0)
        new_agent, critic_info = jax.lax.fori_loop(1, utd_ratio, body, (new_agent, critic_info))

        new_agent, actor_info = new_agent.update_actor(actor_batch)
        if self.ne_samples + self.ne_samples_train > 0:
            new_agent, edit_info = new_agent.update_edit_actor(mini_batch_last)
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            actor_info.update(edit_info)
            actor_info.update(temp_info)
        filter_noise_info = {}
        return new_agent, {**actor_info, **critic_info, **filter_noise_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        assert utd_ratio > 0
        assert batch["observations"].shape[0] % utd_ratio == 0
        mini_batch_size = batch["observations"].shape[0] // utd_ratio

        def get_mini_batch(i):
            start = i * mini_batch_size
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mini_batch_size, axis=0), batch)

        mini_batch_last = get_mini_batch(utd_ratio - 1)

        def body(i, carry):
            agent, _ = carry
            mini_batch = get_mini_batch(i)
            agent, info = agent.update_critic(mini_batch)
            return agent, info

        mini_batch0 = get_mini_batch(0)
        new_agent, critic_info = self.update_critic(mini_batch0)
        new_agent, critic_info = jax.lax.fori_loop(1, utd_ratio, body, (new_agent, critic_info))

        new_agent, actor_info = new_agent.update_actor(mini_batch_last)
        if self.ne_samples + self.ne_samples_train > 0:
            new_agent, edit_info = new_agent.update_edit_actor(mini_batch_last)
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            actor_info.update(edit_info)
            actor_info.update(temp_info)
        filter_noise_info = {}
        return new_agent, {**actor_info, **critic_info, **filter_noise_info}


def get_config():
    from configs import base_config

    config = base_config.get_config()

    config.model_cls = "BetterDiffusionSACLearner"

    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.init_temperature = 1.0
    config.discount = 0.99
    config.tau = 0.005
    config.hidden_dims = (256, 256, 256)
    config.critic_hidden_dims = config_dict.placeholder(tuple)
    config.outer_critic_hidden_dims = config_dict.placeholder(tuple)
    config.filter_critic_hidden_dims = config_dict.placeholder(tuple)
    config.critic_dropout_rate = None
    config.critic_weight_decay = None
    config.use_pnorm = False
    config.use_critic_resnet = False

    config.num_qs = 10
    config.num_min_qs = 2
    config.filter_num_min_qs = 2
    config.critic_layer_norm = True

    config.T = 10
    config.time_dim = 128
    config.N = 8
    config.train_N = 8
    config.M = 0
    config.ddim_eta = 0.0
    config.beta_schedule = "vp"

    config.filter_enabled = True
    config.filter_at_eval = True
    config.filter_temperature_train = 1.0
    config.filter_temperature_eval = 0.0
    config.filter_temperature_mode = "zscore"

    config.ne_samples = 1
    config.ne_samples_train = 1
    config.r_action_scale = 0.15

    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.actor_layer_norm = True
    config.actor_tau = 0.001
    config.actor_num_blocks = 3
    config.decay_steps = int(3e6)

    return config
