import jax
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution, validate_args=False):
        super().__init__(distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def _log_prob(self, value, **kwargs) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=self.distribution.loc.dtype)
        eps = jnp.finfo(value.dtype).eps
        clipped = jnp.clip(value, -1 + eps, 1 - eps)
        pre_tanh = jnp.arctanh(clipped)
        scale_diag = self.distribution.scale.diag
        centered = (pre_tanh - self.distribution.loc) / scale_diag
        base_log_prob = jnp.sum(-0.5 * jnp.square(centered) - jnp.log(scale_diag) - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
        log_det = 2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
        return base_log_prob - jnp.sum(log_det, axis=-1)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
