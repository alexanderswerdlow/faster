from functools import partial
from typing import Callable, Optional, Sequence, Type
import flax.linen as nn
import jax.numpy as jnp
import jax

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training)


@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_train_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, training = False):

    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis = 1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                            shape=(observations.shape[0], current_x.shape[1]),)
        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = fn(input_tuple, 0)
    
    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng



@partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'N', 'sar_N'))
def ddpm_hidden_train_sampler(actor_apply_fn, actor_params, critic_apply_fn, critic_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, N, sar_N, training = False):

    total_batch_size = observations.shape[0]  # batch_size * N
    batch_size = total_batch_size // N
    
    # Denoise first step
    key, rng = jax.random.split(rng, 2)
    current_x = jax.random.normal(key, (total_batch_size, act_dim))
    
    # First step (time = T-1)
    time = T - 1
    input_time = jnp.full((total_batch_size, 1), time)
    eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training=training)
    
    alpha_1 = 1 / jnp.sqrt(alphas[time])
    alpha_2 = (1 - alphas[time]) / jnp.sqrt(1 - alpha_hats[time])
    first_step_hidden = alpha_1 * (current_x - alpha_2 * eps_pred)
    
    # Add noise
    key, rng = jax.random.split(rng, 2)
    z = jax.random.normal(key, shape=(total_batch_size, act_dim))
    z_scaled = sample_temperature * z
    current_x = first_step_hidden + jnp.sqrt(betas[time]) * z_scaled
    
    if clip_sampler:
        current_x = jnp.clip(current_x, -1, 1)
    
    # Evaluate and filter
    critic_values = critic_apply_fn({"params": critic_params}, observations, first_step_hidden)
    
    
    # Reshape to (batch_size, N)
    critic_values_reshaped = critic_values.min(axis=0).reshape(batch_size, N)
    current_x_reshaped = current_x.reshape(batch_size, N, act_dim)
    first_step_hidden_reshaped = first_step_hidden.reshape(batch_size, N, act_dim)
    observations_reshaped = observations.reshape(batch_size, N, -1)
    
    # Get top M indices
    _, top_m_indices = jax.lax.top_k(critic_values_reshaped, sar_N)
    batch_indices = jnp.arange(batch_size)[:, None]
    
    # Filter to top M
    filtered_x = current_x_reshaped[batch_indices, top_m_indices].reshape(batch_size * sar_N, act_dim)
    filtered_observations = observations_reshaped[batch_indices, top_m_indices].reshape(batch_size * sar_N, -1)
    filtered_first_step = first_step_hidden_reshaped[batch_indices, top_m_indices].reshape(batch_size * sar_N, act_dim)
    filtered_critic_values = critic_values_reshaped[batch_indices, top_m_indices].reshape(batch_size * sar_N)
    
    # Continue denoising
    def fn(current_x, time):
        input_time = jnp.full((current_x.shape[0], 1), time)
        eps_pred = actor_apply_fn({"params": actor_params}, filtered_observations, current_x, input_time, training=training)
        
        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = (1 - alphas[time]) / jnp.sqrt(1 - alpha_hats[time])
        current_x_denoised = alpha_1 * (current_x - alpha_2 * eps_pred)
        
        key_t = jax.random.fold_in(rng, time)
        z = jax.random.normal(key_t, shape=current_x.shape)
        z_scaled = sample_temperature * z
        current_x = current_x_denoised + (time > 0) * jnp.sqrt(betas[time]) * z_scaled
        
        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)
        
        return current_x, ()
    
    # Run remaining T-2 steps (we already did step T-1)
    if T > 1:
        filtered_x, () = jax.lax.scan(fn, filtered_x, jnp.arange(T-2, -1, -1))
    
    # Repeat last step
    for _ in range(repeat_last_step):
        filtered_x, () = fn(filtered_x, 0)
    
    action_0 = jnp.clip(filtered_x, -1, 1)
    
    return action_0, rng, filtered_first_step, filtered_critic_values




# @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'N', 'sar_N'))
# def ddpm_hidden_train_sampler(actor_apply_fn, actor_params, critic_apply_fn, critic_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, N, sar_N, training = False):

#     total_batch_size = observations.shape[0]  # batch_size * N
#     batch_size = total_batch_size // N
    
#     def fn(input_tuple, time):
#         current_x, rng, first_step_hidden = input_tuple
        
#         input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis = 1)
#         eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

#         alpha_1 = 1 / jnp.sqrt(alphas[time])
#         alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
#         current_x_denoised = alpha_1 * (current_x - alpha_2 * eps_pred)

#         # Store hidden values from first denoising step (time == T-1)
#         first_step_hidden = jax.lax.cond(
#             time == T - 1,
#             lambda: current_x_denoised,
#             lambda: first_step_hidden
#         )
        
#         rng, key = jax.random.split(rng, 2)
#         z = jax.random.normal(key,
#                             shape=(observations.shape[0], current_x_denoised.shape[1]),)
#         z_scaled = sample_temperature * z
#         current_x = current_x_denoised + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

#         if clip_sampler:
#             current_x = jnp.clip(current_x, -1, 1)

#         return (current_x, rng, first_step_hidden), ()

#     key, rng = jax.random.split(rng, 2)
#     initial_noise = jax.random.normal(key, (total_batch_size, act_dim))
#     initial_first_step = jnp.zeros_like(initial_noise)
    
#     # Run denoising up to first step
#     input_tuple, () = jax.lax.scan(
#         fn, 
#         (initial_noise, rng, initial_first_step), 
#         jnp.arange(T-1, -1, -1)
#     )
    
#     current_x, rng, first_step_hidden = input_tuple

#     # Evaluate all samples with critic
#     critic_values = critic_apply_fn({"params": critic_params}, observations, first_step_hidden)
#     # critic_values shape: (batch_size * N, 1) or (batch_size * N,)
#     critic_values = critic_values.squeeze()  # Shape: (batch_size * N,)
    
#     # Reshape to (batch_size, N) to filter within each batch element
#     critic_values_reshaped = critic_values.min(axis=0).reshape(batch_size, N)
#     current_x_reshaped = current_x.reshape(batch_size, N, act_dim)
#     first_step_hidden_reshaped = first_step_hidden.reshape(batch_size, N, act_dim)
#     observations_reshaped = observations.reshape(batch_size, N, -1)
    
#     # Get top M indices for each batch element
#     top_m_indices = jnp.argsort(critic_values_reshaped, axis=1)[:, -sar_N:]  # Shape: (batch_size, M)
    
#     # Gather top M samples
#     batch_indices = jnp.arange(batch_size)[:, None]  # Shape: (batch_size, 1)
    
#     filtered_x = current_x_reshaped[batch_indices, top_m_indices]  # Shape: (batch_size, M, act_dim)
#     filtered_observations = observations_reshaped[batch_indices, top_m_indices]  # Shape: (batch_size, M, obs_dim)
#     filtered_first_step = first_step_hidden_reshaped[batch_indices, top_m_indices]  # Shape: (batch_size, M, act_dim)
#     filtered_critic_values = critic_values_reshaped[batch_indices, top_m_indices]  # Shape: (batch_size, M)
    
#     # Flatten back to (batch_size * M, ...)
#     filtered_x = filtered_x.reshape(batch_size * sar_N, act_dim)
#     filtered_observations = filtered_observations.reshape(batch_size * sar_N, -1)
#     filtered_first_step = filtered_first_step.reshape(batch_size * sar_N, act_dim)
#     filtered_critic_values = filtered_critic_values.reshape(batch_size * sar_N)

#     # Continue denoising with only the filtered samples
#     for _ in range(repeat_last_step):
#         input_time = jnp.expand_dims(jnp.array([0]).repeat(filtered_x.shape[0]), axis = 1)
#         eps_pred = actor_apply_fn({"params": actor_params}, filtered_observations, filtered_x, input_time, training = training)
        
#         alpha_1 = 1 / jnp.sqrt(alphas[0])
#         alpha_2 = ((1 - alphas[0]) / (jnp.sqrt(1 - alpha_hats[0])))
#         filtered_x = alpha_1 * (filtered_x - alpha_2 * eps_pred)
        
#         if clip_sampler:
#             filtered_x = jnp.clip(filtered_x, -1, 1)
    
#     action_0 = jnp.clip(filtered_x, -1, 1)

#     return action_0, rng, filtered_first_step, filtered_critic_values
    

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'clip_sampler', 'training'))
def ddpm_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, training = False):

    batch_size = observations.shape[0]
    init_key, rng = jax.random.split(rng)
    noise_key, rng = jax.random.split(rng)

    def step(current_x, time):
        input_time = jnp.full((current_x.shape[0], 1), time)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        z = jax.random.normal(jax.random.fold_in(noise_key, time), shape=current_x.shape)
        noise_scale = jnp.where(time > 0, jnp.sqrt(betas[time]) * sample_temperature, 0.0)
        current_x = current_x + noise_scale * z

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return current_x, ()

    current_x = jax.random.normal(init_key, (batch_size, act_dim))
    current_x, () = jax.lax.scan(step, current_x, jnp.arange(T - 1, -1, -1))

    def repeat_body(_, x):
        x, () = step(x, 0)
        return x

    current_x = jax.lax.fori_loop(0, repeat_last_step, repeat_body, current_x)
    action_0 = jnp.clip(current_x, -1, 1)

    return action_0, rng


@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'clip_sampler', 'training'))
def ddim_sampler(
    actor_apply_fn,
    actor_params,
    T,
    rng,
    act_dim,
    observations,
    alphas,
    alpha_hats,
    betas,
    sample_temperature,
    repeat_last_step,
    clip_sampler,
    training=False,
    *,
    eta: float = 0.0,
):

    batch_size = observations.shape[0]
    init_key, rng = jax.random.split(rng)
    noise_key, rng = jax.random.split(rng)

    def step(current_x, time):
        input_time = jnp.full((current_x.shape[0], 1), time)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

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
            x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + jnp.sqrt(1.0 - alpha_hat_prev) * eps_pred
            return x_next

        def stochastic(_):
            sigma = eta_ * jnp.sqrt(
                (1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev)
            )
            z = jax.random.normal(jax.random.fold_in(noise_key, time), shape=current_x.shape)
            noise = sigma * sample_temperature * z
            eps_scale = jnp.sqrt(jnp.maximum(0.0, 1.0 - alpha_hat_prev - sigma**2))
            x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + eps_scale * eps_pred + noise
            return x_next

        current_x = jax.lax.cond(eta_ == 0.0, deterministic, stochastic, operand=None)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return current_x, ()

    current_x = jax.random.normal(init_key, (batch_size, act_dim))
    current_x, () = jax.lax.scan(step, current_x, jnp.arange(T - 1, -1, -1))

    def repeat_body(_, x):
        x, () = step(x, 0)
        return x

    current_x = jax.lax.fori_loop(0, repeat_last_step, repeat_body, current_x)
    action_0 = jnp.clip(current_x, -1, 1)

    return action_0, rng


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params


default_init = nn.initializers.xavier_uniform

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

class DiffusionMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class MultiHeadStateActionValue(nn.Module):
    base_cls: nn.Module
    num_heads: int

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        head_outputs = []
        for i in range(self.num_heads):
            head_output = nn.Dense(1, kernel_init=default_init())(outputs)
            head_outputs.append(jnp.squeeze(head_output, -1))


        # value = nn.Dense(1, kernel_init=default_init())(outputs)

        # return jnp.squeeze(value, -1)

        return head_outputs


default_init = nn.initializers.xavier_uniform

class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x

class DiffusionMLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
            
        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x
