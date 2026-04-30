from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from faster.agents.agent import Agent
from faster.data.dataset import DatasetDict
from faster.networks import (
    DDPM,
    MLP,
    DiffusionMLP,
    DiffusionMLPResNet,
    Ensemble,
    FourierFeatures,
    MLPResNetV2,
    StateActionValue,
    StateValue,
    cosine_beta_schedule,
    ddim_sampler,
    ddpm_sampler,
    vp_beta_schedule,
)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=("critic_fn"))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values


@jax.jit
def _sample_actions_jit(agent, observations):
    rng = agent.rng
    observations = jnp.squeeze(observations)
    assert observations.ndim == 1, observations.shape
    observations = jnp.expand_dims(observations, axis=0).repeat(agent.N, axis=0)

    actor_params = agent.target_actor.params
    if agent.deterministic_ddim_eta0:
        actions, rng = ddim_sampler(
            agent.actor.apply_fn,
            actor_params,
            agent.T,
            rng,
            agent.action_dim,
            observations,
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.M,
            eta=agent.ddim_eta,
        )
    else:
        actions, rng = ddpm_sampler(
            agent.actor.apply_fn,
            actor_params,
            agent.T,
            rng,
            agent.action_dim,
            observations,
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.ddpm_temperature,
            agent.M,
            True,
        )

    qs = compute_q(agent.target_critic.apply_fn, agent.target_critic.params, observations, actions)
    action = actions[jnp.argmax(qs)]
    rng, _ = jax.random.split(rng, 2)
    return action, rng


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IDQLLearner(Agent):
    critic: TrainState
    value: TrainState
    target_critic: TrainState
    target_actor: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    expectile: float
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddpm_temperature: float
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    deterministic_ddim_eta0: bool = struct.field(pytree_node=False)
    ddim_eta: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)  # See M in RedQ https://arxiv.org/abs/2101.05982

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
        M: int = 0,
        ng_samples: int = 0,
        ng_samples_train: int = 0,
        nr_samples: int = 0,
        nr_samples_train: int = 0,
        r_action_scale: float = 1.0,
        on_policy_residual: int = 0,
        actor_layer_norm: bool = True,
        decay_steps: Optional[int] = int(3e6),
        deterministic_ddim_eta0: bool = False,
        ddim_eta: float = 0.0,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = "vp",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]

        if isinstance(action_space, gym.Space):
            observations = observation_space.sample()
            actions = action_space.sample()

        else:
            observations = observation_space
            actions = action_space

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        """

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        """

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
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
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
            tx = optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay, mask=decay_mask_fn)
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=tx)
        target_critic_def = Ensemble(critic_cls, num=num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
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
            tx = optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay, mask=decay_mask_fn)
        else:
            tx = optax.adam(learning_rate=critic_lr)

        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=tx)

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            expectile=expectile,
            action_dim=action_dim,
            T=T,
            N=N,
            M=M,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            value=value,
            critic=critic,
            target_critic=target_critic,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            deterministic_ddim_eta0=deterministic_ddim_eta0,
            ddim_eta=ddim_eta,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
        )

    def eval_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def sample_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch["actions"].shape[0],), 0, self.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (batch["actions"].shape[0], self.action_dim))

        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = self.actor.apply_fn(
                {"params": score_model_params}, batch["observations"], noisy_actions, time, rngs={"dropout": key}, training=True
            )

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return actor_loss, {"actor_loss": actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        agent = self.replace(actor=actor)
        target_score_params = optax.incremental_update(actor.params, self.target_actor.params, self.actor_tau)

        target_score_model = self.target_actor.replace(params=target_score_params)
        new_agent = self.replace(actor=actor, target_actor=target_score_model, rng=rng)

        return new_agent, info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_value(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)
        qs = self.target_critic.apply_fn(
            {"params": self.target_critic.params}, batch["observations"], batch["actions"], True, rngs={"dropout": key}
        )
        q = qs.min(axis=0)

        key, rng = jax.random.split(rng)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = self.value.apply_fn({"params": value_params}, batch["observations"], True, rngs={"dropout": key})
            value_loss = expectile_loss(q - v, self.expectile).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params)
        value = self.value.apply_gradients(grads=grads)
        return self.replace(value=value, rng=rng), info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        # dist = self.actor.apply_fn(
        #     {"params": self.actor.params}, batch["next_observations"]
        # )

        # rng = self.rng

        # key, rng = jax.random.split(rng)
        # next_actions = dist.sample(seed=key)

        rng = self.rng

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        # target_params = subsample_ensemble(
        #     key, self.target_critic.params, self.num_min_qs, self.num_qs
        # )

        key, rng = jax.random.split(rng)
        next_qs = self.value.apply_fn(
            {"params": self.value.params}, batch["next_observations"], True, rngs={"dropout": key}
        )  # training=True
        next_q = next_qs

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"], True, rngs={"dropout": key}
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_r"))
    def update_offline(self, batch: DatasetDict, utd_ratio: int, pretrain_q: bool, pretrain_r: bool):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            critic_info = {}
            new_agent, value_info = new_agent.update_value(mini_batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        return new_agent, {**actor_info, **critic_info, **value_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_separate(self, batch: DatasetDict, actor_batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(actor_batch)

        if self.ng_samples + self.ng_samples_train > 0:
            new_agent, actor_info = new_agent.update_gaussian_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

            actor_info.update(temp_info)

        if self.nr_samples + self.nr_samples_train > 0:
            new_agent, actor_info = new_agent.update_residual_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

            actor_info.update(temp_info)

        return new_agent, {**actor_info, **critic_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, value_info = new_agent.update_value(mini_batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        return new_agent, {**actor_info, **value_info, **critic_info}


def get_config():
    from configs import base_config

    config = base_config.get_config()
    config.model_cls = "IDQLLearner"
    config.num_qs = 2
    config.num_min_qs = 1
    config.critic_layer_norm = True
    config.expectile = 0.8
    config.N = 8
    config.train_N = 8
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.ddim_eta = 0.0
    config.T = 10
    return config
