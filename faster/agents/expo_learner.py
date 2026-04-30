"""Implementations of algorithms for continuous control."""

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
from faster.agents.temperature import Temperature
from faster.data.dataset import DatasetDict
from faster.distributions import TanhNormal
from faster.networks import (
    DDPM,
    MLP,
    DiffusionMLP,
    DiffusionMLPResNet,
    Ensemble,
    FourierFeatures,
    MLPResNetV2,
    StateActionValue,
    cosine_beta_schedule,
    ddpm_sampler,
    ddpm_train_sampler,
    subsample_ensemble,
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


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


def sample_from_probs(key, probs):
    return jax.random.choice(key, len(probs), p=probs)


class EXPOLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    target_actor: TrainState
    edit_actor: TrainState
    temp: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    clip_sampler: bool = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    train_N: int = struct.field(pytree_node=False)
    ne_samples: int = struct.field(pytree_node=False)
    ne_samples_train: int = struct.field(pytree_node=False)
    r_action_scale: float = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddpm_temperature: float
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    soft_sampling_beta: float = struct.field(pytree_node=False)
    soft_sampling_dist_backup: bool = struct.field(pytree_node=False)
    soft_sampling_dist: bool = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        soft_sampling_dist: bool = False,
        soft_sampling_dist_backup: bool = False,
        soft_sampling_beta: float = 1.0,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None,
        d_actor_drop: Optional[float] = None,
        T: int = 10,
        N: int = 32,
        train_N: int = 32,
        batch_split: int = 1,
        M: int = 0,
        ne_samples: int = 0,
        ne_samples_train: int = 0,
        r_action_scale: float = 1.0,
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
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

        edit_actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm)
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate([observations, jnp.ones((1, action_dim))], axis=1)
        edit_actor_params = edit_actor_def.init(actor_key, edit_observations)["params"]
        edit_actor = TrainState.create(apply_fn=edit_actor_def.apply, params=edit_actor_params, tx=optax.adam(learning_rate=actor_lr))

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
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr))

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            edit_actor=edit_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            action_dim=action_dim,
            clip_sampler=clip_sampler,
            T=T,
            soft_sampling_dist_backup=soft_sampling_dist_backup,
            soft_sampling_dist=soft_sampling_dist,
            soft_sampling_beta=soft_sampling_beta,
            N=N,
            train_N=train_N,
            ne_samples=ne_samples,
            ne_samples_train=ne_samples_train,
            r_action_scale=r_action_scale,
            batch_split=batch_split,
            M=M,
            actor_tau=actor_tau,
            ddpm_temperature=ddpm_temperature,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    def eval_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)

        actor_params = self.target_actor.params
        actions, rng = ddpm_sampler(
            self.actor.apply_fn,
            actor_params,
            self.T,
            rng,
            self.action_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
        )

        diffusion_actions = actions

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

            if self.ne_samples > 0:
                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate(
                    [observations, jnp.expand_dims(observations[0], axis=0).repeat(self.ne_samples, axis=0)], axis=0
                )

                r_observations = jnp.repeat(jnp.expand_dims(observations[0], axis=0), self.ne_samples, axis=0)
                d_actions = diffusion_actions.copy()[: self.ne_samples]
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng = _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
                r_samples = r_samples * self.r_action_scale + d_actions
                actions = jnp.concatenate([actions, r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)

            if self.soft_sampling_dist:
                soft_qs = jax.nn.softmax(self.soft_sampling_beta * qs)

                key, rng = jax.random.split(rng, 2)
                idx = jax.random.choice(key, len(soft_qs), p=soft_qs)

            else:
                idx = jnp.argmax(qs)
            action = actions[idx]

        else:
            action = actions[0]

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def sample_batch_actions(self, observations):
        rng = self.rng

        observations = jnp.squeeze(observations)
        observations = jax.device_put(observations)

        batch_size = observations.shape[0]

        # Repeat each observation N times: (batch_size, obs_dim) -> (batch_size * N, obs_dim)
        observations_repeated = jnp.repeat(observations, self.train_N, axis=0)

        actor_params = self.actor.params
        actions_flat, rng = ddpm_train_sampler(
            self.actor.apply_fn,
            actor_params,
            self.T,
            rng,
            self.action_dim,
            observations_repeated,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
        )

        # Reshape actions from (batch_size * N, action_dim) to (batch_size, N, action_dim)
        actions = actions_flat.reshape(batch_size, self.train_N, -1)

        observations_repeated = jnp.repeat(observations, self.train_N + self.ne_samples_train, axis=0)

        if self.ne_samples_train > 0:
            key, rng = jax.random.split(rng, 2)
            r_observations = jnp.repeat(observations, self.ne_samples_train, axis=0)  # (batch_size * ne_samples_train, obs_dim) repeats
            d_actions = actions.copy()[:, : self.ne_samples_train].reshape(-1, actions.shape[-1])
            r_observations = jnp.concatenate([r_observations, d_actions], axis=1)  # self.ne_samples_train actions for each observation
            r_samples, rng = _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
            # actions_flat = jnp.concatenate([actions_flat.copy(), r_samples])
            r_samples = r_samples * self.r_action_scale + d_actions
            actions = jnp.concatenate([actions, r_samples.reshape(batch_size, self.ne_samples_train, -1)], axis=1)
            actions_flat = actions.reshape(-1, actions.shape[-1])

        if self.train_N > 1:
            # Reshape observations back to (batch_size, N, obs_dim) for Q computation
            # observations_for_q = observations_repeated.reshape(batch_size, self.N, -1)

            # Compute Q-values for all action samples
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

            # Flatten for Q computation: (batch_size * N, obs_dim), (batch_size * N, action_dim)
            # obs_flat = observations_for_q.reshape(-1, observations_for_q.shape[-1])
            # actions_flat = actions.reshape(-1, actions.shape[-1])
            obs_flat = observations_repeated
            actions_flat = actions_flat

            # Compute Q-values: (batch_size * N,)

            if self.batch_split > 1:
                q_list = []

                one_call = (batch_size * (self.train_N + self.ne_samples_train)) // self.batch_split

                for i in range(self.batch_split):
                    obs_i = obs_flat[i * one_call : (i + 1) * one_call]
                    actions_i = actions_flat[i * one_call : (i + 1) * one_call]

                    q_list += [compute_q(self.target_critic.apply_fn, target_params, obs_i, actions_i)]

                    # qs[i * self.train_N : (i + 1) * self.train_N] = compute_q(self.target_critic.apply_fn, target_params, obs_i, actions_i)

                qs = jnp.concatenate(q_list)

            else:
                qs = compute_q(self.target_critic.apply_fn, target_params, obs_flat, actions_flat)

            # Reshape Q-values: (batch_size, N)
            qs = qs.reshape(batch_size, self.train_N + self.ne_samples_train)

            if self.soft_sampling_dist_backup:
                soft_qs = jax.nn.softmax(self.soft_sampling_beta * qs, axis=1)

                keys = jax.random.split(key, qs.shape[0])

                best_indices = jax.vmap(sample_from_probs)(keys, soft_qs)

            else:
                # Select best action for each observation
                best_indices = jnp.argmax(qs, axis=1)  # (batch_size,)

            # Use advanced indexing to select best actions
            batch_indices = jnp.arange(batch_size)
            best_actions = actions[batch_indices, best_indices]  # (batch_size, action_dim)
        else:
            # If N=1, just take the single action for each observation
            best_actions = actions[:, 0]  # (batch_size, action_dim)

        rng, _ = jax.random.split(rng, 2)
        return jnp.array(best_actions.squeeze())

    def sample_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)

        actor_params = self.target_actor.params
        actions, rng = ddpm_sampler(
            self.actor.apply_fn,
            actor_params,
            self.T,
            rng,
            self.action_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.M,
            self.clip_sampler,
        )

        diffusion_actions = actions

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

            if self.ne_samples > 0:
                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate(
                    [observations, jnp.expand_dims(observations[0], axis=0).repeat(self.ne_samples, axis=0)], axis=0
                )

                r_observations = jnp.repeat(jnp.expand_dims(observations[0], axis=0), self.ne_samples, axis=0)
                d_actions = diffusion_actions.copy()[: self.ne_samples]
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng = _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
                r_samples = r_samples * self.r_action_scale + d_actions
                actions = jnp.concatenate([actions, r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)

            if self.soft_sampling_dist:
                soft_qs = jax.nn.softmax(self.soft_sampling_beta * qs)

                key, rng = jax.random.split(rng, 2)
                idx = jax.random.choice(key, len(soft_qs), p=soft_qs)

            else:
                idx = jnp.argmax(qs)

            action = actions[idx]

        else:
            action = actions[0]

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def update_edit_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            edit_observations = jnp.concatenate([batch["observations"], batch["actions"]], axis=1)
            dist = self.edit_actor.apply_fn({"params": actor_params}, edit_observations, training=True, rngs={"dropout": dropout_key})
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)

            actions = actions * self.r_action_scale

            # Subtract log of action scale for each action dimension
            log_probs -= actions.shape[-1] * jnp.log(self.r_action_scale)

            actions += batch["actions"]

            qs = self.critic.apply_fn(
                {"params": self.critic.params}, batch["observations"], actions, True, rngs={"dropout": key2}
            )  # training=True
            q = qs.mean(axis=0)
            edit_actor_loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info

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

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        next_actions = self.sample_batch_actions(batch["next_observations"])

        rng = self.rng

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params}, batch["next_observations"], next_actions, True, rngs={"dropout": key}
        )  # training=True
        next_q = next_qs.min(axis=0)

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
            if pretrain_q:
                new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        if pretrain_r:
            if self.ne_samples + self.ne_samples_train > 0:
                new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
                new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

                actor_info.update(temp_info)

        return new_agent, {**actor_info, **critic_info}

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

        if self.ne_samples + self.ne_samples_train > 0:
            new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
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
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        if self.ne_samples + self.ne_samples_train > 0:
            new_agent, actor_info = new_agent.update_edit_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

            actor_info.update(temp_info)

        return new_agent, {**actor_info, **critic_info}


def get_config():
    from configs import base_config

    config = base_config.get_config()
    config.model_cls = "EXPOLearner"

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.N = 32
    config.train_N = 32
    config.target_entropy = None
    config.ne_samples = 0
    config.ne_samples_train = 0
    config.adjust_target_entropy = False
    config.soft_sampling_dist_backup = False
    config.soft_sampling_dist = False
    config.soft_sampling_beta = 1.0
    config.r_action_scale = 1.0
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.batch_split = 1
    config.T = 10

    config.backup_entropy = False
    config.hidden_dims = (256, 256, 256)
    config.num_min_qs = 2
    config.N = 8
    config.train_N = 8
    config.r_action_scale = 0.15

    return config
