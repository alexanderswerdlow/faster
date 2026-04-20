"""Implementations of algorithms for continuous control."""

import ast
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlpd.agents.agent import Agent
from rlpd.data.dataset import DatasetDict
from rlpd.networks import (
    DDPM,
    DiffusionMLP,
    DiffusionMLPResNet,
    Ensemble,
    FourierFeatures,
    MLP,
    MLPResNetV2,
    StateActionValue,
    StateValue,
    cosine_beta_schedule,
    ddim_sampler,
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


@partial(jax.jit, static_argnames=("critic_fn",))
def compute_q_with_time(critic_fn, critic_params, observations, actions, time):
    q_values = critic_fn({"params": critic_params}, observations, actions, time)
    return q_values.min(axis=0)


@jax.jit
def _sample_actions_jit(agent, observations):
    observations = jnp.squeeze(observations)
    assert observations.ndim == 1, observations.shape
    observations = observations[None, :]

    actions, rng = agent._sample_filtered_diffusion_candidates(observations, agent.N, agent.target_actor.params)
    actions = actions.reshape(-1, agent.action_dim)
    obs_rep = jnp.broadcast_to(observations, (actions.shape[0], observations.shape[-1]))
    qs = compute_q(agent.target_critic.apply_fn, agent.target_critic.params, obs_rep, actions)
    action = actions[jnp.argmax(qs)]
    rng, _ = jax.random.split(rng)
    return action, rng


@partial(
    jax.jit,
    static_argnames=("actor_apply_fn", "act_dim", "T", "repeat_last_step", "clip_sampler", "training", "use_ddim"),
)
def diffusion_sampler_from_x(
    actor_apply_fn,
    actor_params,
    T,
    rng,
    act_dim,
    observations,
    init_x,
    alphas,
    alpha_hats,
    betas,
    sample_temperature,
    repeat_last_step,
    clip_sampler,
    training=False,
    *,
    use_ddim=False,
    eta: float = 0.0,
):
    noise_key, rng = jax.random.split(rng)

    def step(current_x, time):
        input_time = jnp.full((current_x.shape[0], 1), time)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training=training)

        if use_ddim:
            alpha_hat_t = alpha_hats[time]
            sqrt_alpha_hat_t = jnp.sqrt(alpha_hat_t)
            sqrt_one_minus_alpha_hat_t = jnp.sqrt(1.0 - alpha_hat_t)
            x0_pred = (current_x - sqrt_one_minus_alpha_hat_t * eps_pred) / sqrt_alpha_hat_t
            alpha_hat_prev = jax.lax.cond(
                time > 0,
                lambda t: alpha_hats[t - 1],
                lambda _: jnp.asarray(1.0, dtype=alpha_hat_t.dtype),
                time,
            )
            eta_ = jnp.asarray(eta, dtype=current_x.dtype)

            def deterministic(_):
                return jnp.sqrt(alpha_hat_prev) * x0_pred + jnp.sqrt(1.0 - alpha_hat_prev) * eps_pred

            def stochastic(_):
                sigma = eta_ * jnp.sqrt((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev))
                z = jax.random.normal(jax.random.fold_in(noise_key, time), shape=current_x.shape)
                noise = sigma * sample_temperature * z
                eps_scale = jnp.sqrt(jnp.maximum(0.0, 1.0 - alpha_hat_prev - sigma**2))
                return jnp.sqrt(alpha_hat_prev) * x0_pred + eps_scale * eps_pred + noise

            current_x = jax.lax.cond(eta_ == 0.0, deterministic, stochastic, operand=None)
        else:
            alpha_1 = 1.0 / jnp.sqrt(alphas[time])
            alpha_2 = (1.0 - alphas[time]) / jnp.sqrt(1.0 - alpha_hats[time])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
            z = jax.random.normal(jax.random.fold_in(noise_key, time), shape=current_x.shape)
            noise_scale = jnp.where(time > 0, jnp.sqrt(betas[time]) * sample_temperature, 0.0)
            current_x = current_x + noise_scale * z

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return current_x, ()

    current_x, () = jax.lax.scan(step, init_x, jnp.arange(T - 1, -1, -1))

    def repeat_body(_, x):
        x, () = step(x, 0)
        return x

    current_x = jax.lax.fori_loop(0, repeat_last_step, repeat_body, current_x)
    return jnp.clip(current_x, -1, 1), rng


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff**2)


_ALLOWED_FILTER_TEMPERATURE_MODES = ("plain", "zscore")


def _validate_filter_temperature_mode(mode: str) -> None:
    if mode not in _ALLOWED_FILTER_TEMPERATURE_MODES:
        raise ValueError(f"Invalid filter_temperature_mode={mode}. Allowed: {_ALLOWED_FILTER_TEMPERATURE_MODES}")


def _z_score_normalize(values: jnp.ndarray, axis: int, eps: float = 1e-6) -> jnp.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / jnp.maximum(std, eps)


def _gumbel_topk(key, logits, k):
    u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    _, idx = jax.lax.top_k(logits + g, k)
    return idx


def sample_k_indices(key, scores: jnp.ndarray, k: int, *, temperature: float, mode: str = "plain") -> jnp.ndarray:
    _validate_filter_temperature_mode(mode)
    scores = jnp.asarray(scores)
    if scores.ndim < 1:
        raise ValueError(f"scores must have ndim >= 1; got shape={scores.shape}")
    n = scores.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}; got k={k}")

    prefix = scores.shape[:-1]
    batch = int(np.prod(prefix)) if prefix else 1
    scores2 = scores.reshape(batch, n)
    proc = scores2 if mode == "plain" else _z_score_normalize(scores2, axis=1)

    temp = jnp.asarray(temperature, dtype=proc.dtype)
    temp = jnp.broadcast_to(temp, (batch,))
    do_sample = temp > 0
    temp_safe = jnp.where(do_sample, temp, 1.0)
    logits = proc / temp_safe[:, None]

    idx_sample = _gumbel_topk(key, logits, k)
    idx_det = jax.lax.top_k(scores2, k)[1]
    idx = jnp.where(do_sample[:, None], idx_sample, idx_det)
    return idx.reshape(prefix + (k,))


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
        value = nn.Dense(1)(outputs)
        return jnp.squeeze(value, -1)


def _maybe_literal_eval(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _single_value(name: str, value, default, cast):
    value = _maybe_literal_eval(value)
    if value is None:
        return cast(default)
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"{name} must have length 1; got {value}")
        value = value[0]
    return cast(value)


class IDQLLearnerFast(Agent):
    critic: TrainState
    value: TrainState
    target_critic: TrainState
    target_actor: TrainState
    filter_critic: Optional[TrainState]
    target_filter_critic: Optional[TrainState]
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    expectile: float
    clip_sampler: bool = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    train_N: int = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddpm_temperature: float
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    filter_enabled: bool = struct.field(pytree_node=False)
    filter_at_eval: bool = struct.field(pytree_node=False)
    filter_temperature_eval_sampling: float = struct.field(pytree_node=False)
    filter_temperature_mode: str = struct.field(pytree_node=False)
    ddim_eta: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        expectile=0.8,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None,
        d_actor_drop: Optional[float] = None,
        r_alpha: float = 0.0,
        iql_policy: bool = False,
        T: int = 10,
        N: int = 32,
        train_N: int = 32,
        batch_split: int = 1,
        M: int = 0,
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = "vp",
        deterministic_ddim_eta0: bool = True,
        ddim_eta: float = 0.0,
        filter_enabled: Optional[bool] = None,
        filter_at_eval: bool = False,
        filter_timesteps: Optional[Sequence[int]] = None,
        filter_keep_counts: Optional[Sequence[int]] = None,
        filter_keep_counts_train: Optional[Sequence[int]] = None,
        filter_temperature_eval_sampling: Optional[Sequence[float]] = None,
        filter_temperature_mode: str = "plain",
        filter_critic_lr: Optional[float] = None,
        filter_critic_tau: Optional[float] = None,
        filter_critic_count: int = 1,
        filter_critic_td: bool = False,
        filter_critic_regression_mode: str = "sampled_policy",
        filter_noise_actor_enabled: bool = False,
        filter_noise_actor_steps: int = 1,
        filter_noise_actor_q_loss: bool = True,
        filter_noise_actor_bc_loss: bool = True,
        filter_noise_actor_entropy_loss: bool = True,
        filter_noise_actor_lr: Optional[float] = None,
        filter_noise_actor_tau: Optional[float] = None,
        filter_noise_actor_target_entropy: Optional[float] = None,
        filter_noise_actor_temp_lr: Optional[float] = None,
        filter_noise_actor_init_temperature: float = 1.0,
    ):
        action_dim = action_space.shape[-1]
        _validate_filter_temperature_mode(filter_temperature_mode)
        if filter_enabled is None:
            filter_enabled = bool(filter_at_eval)

        filter_timestep = _single_value("filter_timesteps", filter_timesteps, T, int)
        filter_keep_count = _single_value("filter_keep_counts", filter_keep_counts, 1, int)
        if filter_keep_counts_train is None or len(filter_keep_counts_train) == 0:
            filter_keep_count_train = filter_keep_count
        else:
            filter_keep_count_train = _single_value("filter_keep_counts_train", filter_keep_counts_train, filter_keep_count, int)
        filter_temperature_eval_sampling = _single_value("filter_temperature_eval_sampling", filter_temperature_eval_sampling, 0.0, float)
        assert deterministic_ddim_eta0, deterministic_ddim_eta0
        assert filter_timestep == T, (filter_timestep, T)
        assert filter_keep_count == 1, filter_keep_count
        assert filter_keep_count_train == 1, filter_keep_count_train
        assert N >= 1, N
        assert train_N >= 1, train_N
        assert filter_critic_lr is None, filter_critic_lr
        assert filter_critic_tau is None, filter_critic_tau
        assert filter_critic_count == 1, filter_critic_count
        assert not filter_critic_td, filter_critic_td
        assert filter_critic_regression_mode == "sampled_policy", filter_critic_regression_mode
        assert not filter_noise_actor_enabled, filter_noise_actor_enabled

        if isinstance(action_space, gym.Space):
            observations = observation_space.sample()
            actions = action_space.sample()
        else:
            observations = observation_space
            actions = action_space

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, filter_key = jax.random.split(rng, 5)

        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(
            DiffusionMLP,
            hidden_dims=(time_dim * 2, time_dim * 2),
            activations=nn.swish,
            activate_final=False,
        )

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
        actor_def = DDPM(
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls,
        )

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        actor_params = actor_def.init(actor_key, observations, actions, time)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(learning_rate=actor_lr),
        )
        target_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
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

        if use_critic_resnet:
            critic_base_cls = partial(MLPResNetV2, num_blocks=1)
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            critic_tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            critic_tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=critic_tx)
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
        )
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        if critic_weight_decay is not None:
            value_tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            value_tx = optax.adam(learning_rate=critic_lr)
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=value_tx)

        filter_critic = None
        target_filter_critic = None
        if filter_enabled:
            filter_base_cls = critic_base_cls
            filter_cls = partial(
                DenoisingStateActionValue,
                base_cls=filter_base_cls,
                cond_encoder_cls=cond_model_cls,
                time_preprocess_cls=preprocess_time_cls,
            )
            filter_def = Ensemble(filter_cls, num=1)
            filter_time = jnp.full((1, 1), T, dtype=jnp.int32)
            filter_params = filter_def.init(filter_key, observations, actions, filter_time)["params"]
            if critic_weight_decay is not None:
                filter_tx = optax.adamw(
                    learning_rate=critic_lr,
                    weight_decay=critic_weight_decay,
                    mask=decay_mask_fn,
                )
            else:
                filter_tx = optax.adam(learning_rate=critic_lr)
            filter_critic = TrainState.create(apply_fn=filter_def.apply, params=filter_params, tx=filter_tx)
            target_filter_critic = TrainState.create(
                apply_fn=filter_def.apply,
                params=filter_params,
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            expectile=expectile,
            action_dim=action_dim,
            clip_sampler=clip_sampler,
            T=T,
            N=N,
            train_N=train_N,
            batch_split=batch_split,
            M=M,
            actor_tau=actor_tau,
            ddpm_temperature=ddpm_temperature,
            value=value,
            critic=critic,
            target_critic=target_critic,
            filter_critic=filter_critic,
            target_filter_critic=target_filter_critic,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            filter_enabled=filter_enabled,
            filter_at_eval=filter_at_eval,
            filter_temperature_eval_sampling=filter_temperature_eval_sampling,
            filter_temperature_mode=filter_temperature_mode,
            ddim_eta=float(ddim_eta),
        )

    def _sample_diffusion_candidates(self, observations, N: int, actor_params):
        observations_repeated = jnp.broadcast_to(
            observations[:, None, :],
            (observations.shape[0], N, observations.shape[-1]),
        ).reshape(-1, observations.shape[-1])
        actions, rng = ddim_sampler(
            self.actor.apply_fn,
            actor_params,
            self.T,
            self.rng,
            self.action_dim,
            observations_repeated,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
            eta=self.ddim_eta,
        )
        return actions.reshape(observations.shape[0], N, -1), rng

    def _sample_filter_seed_candidates(self, observations, N: int, rng):
        rng, seed_key = jax.random.split(rng)
        seed_actions = jax.random.normal(seed_key, (observations.shape[0] * N, self.action_dim))
        return seed_actions.reshape(observations.shape[0], N, -1), rng

    def _select_filter_candidates(self, observations, seed_actions, keep_count: int, temperature: float, rng):
        batch_size, sample_count = seed_actions.shape[:2]
        obs_rep = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, sample_count, observations.shape[-1]),
        )
        if keep_count >= sample_count:
            return obs_rep, seed_actions, rng

        assert self.target_filter_critic is not None, self.target_filter_critic
        time = jnp.full((batch_size * sample_count, 1), self.T, dtype=jnp.int32)
        filter_scores = compute_q_with_time(
            self.target_filter_critic.apply_fn,
            self.target_filter_critic.params,
            obs_rep.reshape(-1, observations.shape[-1]),
            seed_actions.reshape(-1, self.action_dim),
            time,
        ).reshape(batch_size, sample_count)
        rng, select_key = jax.random.split(rng)
        idx = sample_k_indices(
            select_key,
            filter_scores,
            keep_count,
            temperature=temperature,
            mode=self.filter_temperature_mode,
        )
        batch_idx = jnp.arange(batch_size)[:, None]
        return obs_rep[batch_idx, idx], seed_actions[batch_idx, idx], rng

    def _prepare_filter_critic_regression_batch(self, batch: DatasetDict, rng):
        observations = batch["observations"]
        seed_actions, rng = self._sample_filter_seed_candidates(observations, self.train_N, rng)
        filter_observations, filter_actions, rng = self._select_filter_candidates(
            observations,
            seed_actions,
            1,
            0.0,
            rng,
        )
        filter_observations = filter_observations.reshape(-1, observations.shape[-1])
        filter_actions = filter_actions.reshape(-1, self.action_dim)
        outer_critic_actions, rng = diffusion_sampler_from_x(
            self.actor.apply_fn,
            self.target_actor.params,
            self.T,
            rng,
            self.action_dim,
            filter_observations,
            filter_actions,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
            use_ddim=True,
            eta=self.ddim_eta,
        )
        q_targets = compute_q(self.target_critic.apply_fn, self.target_critic.params, filter_observations, outer_critic_actions)
        return filter_observations, filter_actions, q_targets, rng

    def _sample_filtered_diffusion_candidates(self, observations, N: int, actor_params):
        if not self.filter_enabled or not self.filter_at_eval:
            return self._sample_diffusion_candidates(observations, N, actor_params)

        needs_filter = self.target_filter_critic is not None and N > 1
        if not needs_filter:
            return self._sample_diffusion_candidates(observations, N, actor_params)

        seed_actions, rng = self._sample_filter_seed_candidates(observations, N, self.rng)
        obs_rep, seed_actions, rng = self._select_filter_candidates(
            observations,
            seed_actions,
            1,
            self.filter_temperature_eval_sampling,
            rng,
        )
        final_actions, rng = diffusion_sampler_from_x(
            self.actor.apply_fn,
            actor_params,
            self.T,
            rng,
            self.action_dim,
            obs_rep.reshape(-1, observations.shape[-1]),
            seed_actions.reshape(-1, self.action_dim),
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
            use_ddim=True,
            eta=self.ddim_eta,
        )
        return final_actions.reshape(observations.shape[0], seed_actions.shape[1], -1), rng

    def eval_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def sample_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
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
                {"params": score_model_params},
                batch["observations"],
                noisy_actions,
                time,
                rngs={"dropout": key},
                training=True,
            )
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return actor_loss, {"actor_loss": actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        target_actor_params = optax.incremental_update(actor.params, self.target_actor.params, self.actor_tau)
        target_actor = self.target_actor.replace(params=target_actor_params)
        return self.replace(actor=actor, target_actor=target_actor, rng=rng), info

    def update_value(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)
        qs = self.target_critic.apply_fn(
            {"params": self.target_critic.params},
            batch["observations"],
            batch["actions"],
            True,
            rngs={"dropout": key},
        )
        q = qs.min(axis=0)

        key, rng = jax.random.split(rng)

        def value_loss_fn(value_params):
            v = self.value.apply_fn({"params": value_params}, batch["observations"], True, rngs={"dropout": key})
            value_loss = expectile_loss(q - v, self.expectile).mean()
            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params)
        value = self.value.apply_gradients(grads=grads)
        return self.replace(value=value, rng=rng), info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_q = self.value.apply_fn(
            {"params": self.value.params},
            batch["next_observations"],
            True,
            rngs={"dropout": key},
        )
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def update_filter_critic(self, batch: DatasetDict):
        if not self.filter_enabled or self.filter_critic is None or self.target_filter_critic is None:
            return self, {}

        rng = self.rng
        filter_observations, filter_actions, q_targets, rng = self._prepare_filter_critic_regression_batch(batch, rng)
        q_targets = jax.lax.stop_gradient(q_targets)
        time = jnp.full((filter_actions.shape[0], 1), self.T, dtype=jnp.int32)
        rng, drop_key = jax.random.split(rng)

        def filter_loss_fn(filter_params):
            qs = self.filter_critic.apply_fn(
                {"params": filter_params},
                filter_observations,
                filter_actions,
                time,
                True,
                rngs={"dropout": drop_key},
            )
            filter_loss = ((qs - q_targets) ** 2).mean()
            info = {
                "filter_critic_loss": filter_loss,
                "filter_q": qs.min(axis=0).mean(),
                "filter_target_q": q_targets.mean(),
            }
            return filter_loss, info

        grads, info = jax.grad(filter_loss_fn, has_aux=True)(self.filter_critic.params)
        filter_critic = self.filter_critic.apply_gradients(grads=grads)
        target_filter_params = optax.incremental_update(
            filter_critic.params,
            self.target_filter_critic.params,
            self.tau,
        )
        target_filter_critic = self.target_filter_critic.replace(params=target_filter_params)
        return self.replace(filter_critic=filter_critic, target_filter_critic=target_filter_critic, rng=rng), info

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_r"))
    def update_offline(self, batch: DatasetDict, utd_ratio: int, pretrain_q: bool, pretrain_r: bool):
        new_agent = self
        for i in range(utd_ratio):

            def slice_fn(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice_fn, batch)
            new_agent, value_info = new_agent.update_value(mini_batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, filter_info = new_agent.update_filter_critic(mini_batch)
        return new_agent, {**actor_info, **critic_info, **value_info, **filter_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_separate(self, batch: DatasetDict, actor_batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice_fn(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice_fn, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(actor_batch)
        new_agent, filter_info = new_agent.update_filter_critic(actor_batch)
        return new_agent, {**actor_info, **critic_info, **filter_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice_fn(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice_fn, batch)
            new_agent, value_info = new_agent.update_value(mini_batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, filter_info = new_agent.update_filter_critic(mini_batch)
        return new_agent, {**actor_info, **value_info, **critic_info, **filter_info}


def get_config():
    from configs import sac_config

    config = sac_config.get_config()
    config.model_cls = "IDQLLearnerFast"
    config.num_qs = 2
    config.num_min_qs = 1
    config.critic_layer_norm = True
    config.expectile = 0.8
    config.N = 32
    config.train_N = 32
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.batch_split = 1
    config.T = 10
    config.ddim_eta = 0.0
    config.filter_enabled = False
    config.filter_at_eval = False
    config.filter_temperature_eval_sampling = (0.0,)
    config.filter_temperature_mode = "plain"
    config.num_min_qs = 2
    return config
