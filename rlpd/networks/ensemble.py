from typing import Type

from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp


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


def _replace_ensemble_params(params, ens_params):
    assert isinstance(params, (dict, FrozenDict)), f"Unsupported params container type: {type(params)}"
    if isinstance(params, FrozenDict):
        return params.copy(add_or_replace={"Ensemble_0": ens_params})
    params = params.copy()
    params["Ensemble_0"] = ens_params
    return params


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = _replace_ensemble_params(params, ens_params)
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params
