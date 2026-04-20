from collections import defaultdict

import flax
import jax


def _numel(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return 0
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def count_params(params):
    return sum(_numel(x) for x in jax.tree_util.tree_leaves(params) if x is not None)


def count_unique_params(*param_trees):
    seen = set()
    total = 0
    for tree in param_trees:
        if tree is None:
            continue
        for leaf in jax.tree_util.tree_leaves(tree):
            if leaf is None:
                continue
            leaf_id = id(leaf)
            if leaf_id in seen:
                continue
            seen.add(leaf_id)
            total += _numel(leaf)
    return total


def _flatten_params(params):
    if hasattr(params, "flat_state"):
        return {path: leaf.value if hasattr(leaf, "value") else leaf for path, leaf in flax.nnx.to_flat_state(params)}
    if isinstance(params, flax.core.FrozenDict):
        params = flax.core.unfreeze(params)
    return flax.traverse_util.flatten_dict(params, keep_empty_nodes=False)


def _summarize_by_depth(params, depth):
    out = defaultdict(int)
    for path, leaf in _flatten_params(params).items():
        if leaf is None:
            continue
        key = "/".join(str(x) for x in path[:depth])
        out[key] += _numel(leaf)
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


def _format_count(n):
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n:,} ({n / 1e9:.2f}B)"
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1e3:.2f}K)"
    return f"{n:,}"


def _is_critic_head_leaf(path, leaf):
    if any("q_head" in str(p) for p in path):
        return True
    if not path:
        return False
    if path[-1] not in {"kernel", "bias"}:
        return False
    shape = getattr(leaf, "shape", None)
    if not shape:
        return False
    return int(shape[-1]) == 1


def _critic_summary(params, *, max_items=8):
    total = count_params(params)
    head = 0
    flat = _flatten_params(params)
    for path, leaf in flat.items():
        if leaf is None:
            continue
        if _is_critic_head_leaf(path, leaf):
            head += _numel(leaf)
    backbone = total - head
    is_ensemble = any(str(k).startswith("Ensemble") for k in getattr(params, "keys", lambda: [])())

    ensemble_size = None
    if is_ensemble:
        for path, leaf in flat.items():
            if not path or not str(path[0]).startswith("Ensemble"):
                continue
            shape = getattr(leaf, "shape", None)
            if shape and len(shape) > 0:
                ensemble_size = int(shape[0])
                break

    depth = 2 if is_ensemble else 1
    by_mod = _summarize_by_depth(params, depth)
    kept = by_mod[:max_items]
    other = total - sum(v for _, v in kept)
    if other > 0:
        kept = [*kept, ("other", other)]
    return {
        "total": total,
        "backbone": backbone,
        "head": head,
        "ensemble_size": ensemble_size,
        "by_module": kept,
    }


def _generic_summary(params, *, max_items=8):
    total = count_params(params)
    by_mod = _summarize_by_depth(params, 1)
    kept = by_mod[:max_items]
    other = total - sum(v for _, v in kept)
    if other > 0:
        kept = [*kept, ("other", other)]
    return {
        "total": total,
        "by_module": kept,
    }


def _summary_params_for_state(agent, field_name, state):
    params = state.params
    actor_name = {
        "actor_train_state": "actor",
        "edit_actor_train_state": "edit_actor",
        "pi05_train_state": "pi05_actor",
    }.get(field_name)
    if actor_name is None or not hasattr(params, "filter"):
        return params
    actor = getattr(agent, actor_name, None)
    train_config = getattr(actor, "train_config", None)
    trainable_filter = getattr(train_config, "trainable_filter", None)
    if trainable_filter is None:
        return params
    return flax.nnx.filter_state(params, trainable_filter)


def format_agent_param_summary(agent, *, max_items=8):
    if hasattr(jax, "process_index") and jax.process_index() != 0:
        return ""

    fields = getattr(agent, "__dataclass_fields__", {})
    states = {}
    for name in fields:
        val = getattr(agent, name)
        if val is None:
            continue
        params = getattr(val, "params", None)
        if params is None:
            continue
        if name.startswith("target_"):
            continue
        states[name] = val

    summary_params = {name: _summary_params_for_state(agent, name, state) for name, state in states.items()}

    ordered = [
        ("actor", "actor"),
        ("edit_actor", "edit_actor"),
        ("critic", "outer_critic"),
        ("filter_critic", "filter_critic"),
        ("temp", "temperature"),
    ]
    for name in sorted(states):
        if any(name == n for n, _ in ordered):
            continue
        ordered.append((name, name))

    lines = ["Parameter counts (trainable params):"]

    critic_params = summary_params.get("critic") if "critic" in summary_params else None
    filter_params = summary_params.get("filter_critic") if "filter_critic" in summary_params else None
    filter_shared = filter_params is not None and critic_params is not None and filter_params is critic_params

    for field_name, display_name in ordered:
        st = states.get(field_name)
        if st is None:
            if field_name in fields and getattr(agent, field_name) is None:
                lines.append(f"- {display_name}: 0")
            continue
        params = summary_params[field_name]
        if field_name in {"critic", "filter_critic"}:
            s = _critic_summary(params, max_items=max_items)
            suffix = " (shared)" if field_name == "filter_critic" and filter_shared else ""
            lines.append(f"- {display_name}{suffix}: {_format_count(s['total'])}")
            if s["ensemble_size"] is not None and s["ensemble_size"] > 0:
                es = int(s["ensemble_size"])
                pt = (s["total"] // es) if (s["total"] % es == 0) else round(s["total"] / es)
                pb = (s["backbone"] // es) if (s["backbone"] % es == 0) else round(s["backbone"] / es)
                ph = (s["head"] // es) if (s["head"] % es == 0) else round(s["head"] / es)
                lines.append(f"    ensemble: {es}")
                lines.append(f"    per_member: total {_format_count(pt)}, backbone {_format_count(pb)}, ensemble_heads {_format_count(ph)}")
            lines.append(f"    backbone: {_format_count(s['backbone'])}")
            lines.append(f"    ensemble_heads: {_format_count(s['head'])}")
            for k, v in s["by_module"]:
                lines.append(f"    {k}: {_format_count(v)}")
        else:
            s = _generic_summary(params, max_items=max_items)
            lines.append(f"- {display_name}: {_format_count(s['total'])}")
            for k, v in s["by_module"]:
                lines.append(f"    {k}: {_format_count(v)}")

    unique_total = count_unique_params(*summary_params.values())
    lines.append(f"- total_unique: {_format_count(unique_total)}")
    return "\n".join(lines)


def print_agent_param_summary(agent, *, max_items=8):
    s = format_agent_param_summary(agent, max_items=max_items)
    if s:
        print(s)
