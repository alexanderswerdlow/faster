#!/usr/bin/env python
import inspect
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle as pickle
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from robomimic.utils.dataset import SequenceDataset

try:
    from flax.training import checkpoints
except Exception:
    print("Not loading checkpointing functionality.")

from rlpd.agents import BetterDiffusionSACLearner
from rlpd.agents import IDQLLearner
from rlpd.agents import IDQLLearnerFast
from rlpd.agents import SACLearner
from rlpd.agents import SAREXPOLearner
from rlpd.data import RoboReplayBuffer
from rlpd.data.robomimic_datasets import (
    ENV_TO_HORIZON_MAP,
    OBS_KEYS,
    RoboD4RLDataset,
    get_robomimic_env,
    process_robomimic_dataset,
)
from rlpd.evaluation import evaluate_robo
from rlpd.param_utils import print_agent_param_summary
from rlpd.train_robo_env_utils import _resolve_robomimic_dataset_path
from rlpd.utils import (
    CsvLogger,
    _build_gitignore_exclude_fn,
    _build_source_code_include_fn,
    _dedupe_config_overrides,
    robomimic_datasets_root,
)

FLAGS = flags.FLAGS
MODEL_REGISTRY = {
    "BetterDiffusionSACLearner": BetterDiffusionSACLearner,
    "IDQLLearner": IDQLLearner,
    "IDQLLearnerFast": IDQLLearnerFast,
    "SACLearner": SACLearner,
    "SAREXPOLearner": SAREXPOLearner,
}

flags.DEFINE_string("project_name", "sample_rank", "wandb project name.")
flags.DEFINE_string("wandb_entity", None, "wandb entity.")
flags.DEFINE_string("wandb_run_group", "", "wandb run group.")
flags.DEFINE_list("wandb_tags", [], "Comma-separated wandb tags.")
flags.DEFINE_boolean("wandb_log_code", True, "Log source code to wandb.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 25000, "Eval interval.")
flags.DEFINE_integer("offline_eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("max_iter", int(1e6), "Number of training iterations.")
flags.DEFINE_integer("start_training", int(1e4), "Number of training steps to start training.")
flags.DEFINE_integer("trajs_per_update", 1, "Number of complete trajectories to collect before each policy update phase.")
flags.DEFINE_integer("grad_updates_per_iter", 1, "Number of gradient updates per iteration.")
flags.DEFINE_integer("num_data", 0, "Number of training steps to start training.")
flags.DEFINE_string("dataset_dir", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("checkpoint_keep", 20, "Number of model checkpoints to keep.")
flags.DEFINE_boolean(
    "skip_initial_eval",
    True,
    "Log synthetic eval metrics at t=0 instead of running a real eval.",
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean("binary_include_bc", True, "Whether to include BC data in the binary datasets.")
flags.DEFINE_boolean("diffusion", False, "Whether to include BC data in the binary datasets.")
flags.DEFINE_boolean("pretrain_r", True, "Whether to include BC data in the binary datasets.")
flags.DEFINE_boolean("pretrain_q", True, "Whether to include BC data in the binary datasets.")
config_flags.DEFINE_config_file("config", "configs/sac_config.py", "File path to the training hyperparameter configuration.", lock_config=False)


def _batch_size(tree):
    if isinstance(tree, dict):
        first_key = next(iter(tree))
        return _batch_size(tree[first_key])
    return tree.shape[0]


def _combine_with_indices(one_tree, other_tree, shuffle_indices):
    combined = {}
    for k, v in one_tree.items():
        if isinstance(v, dict):
            combined[k] = _combine_with_indices(v, other_tree[k], shuffle_indices)
        else:
            other_v = other_tree[k]
            tmp = np.empty((v.shape[0] + other_v.shape[0], *v.shape[1:]), dtype=v.dtype)
            tmp[0 : v.shape[0]] = v
            tmp[v.shape[0] :] = other_v
            combined[k] = np.take(tmp, shuffle_indices, axis=0)
    return combined


def combine(one_dict, other_dict, rng):
    shuffle_indices = rng.permutation(_batch_size(one_dict) + _batch_size(other_dict))
    return _combine_with_indices(one_dict, other_dict, shuffle_indices)


def combine_half(one_dict, other_dict, rng):
    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine_half(v, other_dict[k], rng)
        else:
            other_v = other_dict[k]
            tmp = np.empty((v.shape[0] + other_v.shape[0], *v.shape[1:]), dtype=v.dtype)
            tmp[0::2] = v
            tmp[1::2] = other_v
            combined[k] = tmp
    return combined


def maybe_evaluate_robo(agent, env, max_traj_len, num_episodes, step, save_video=False):
    if FLAGS.skip_initial_eval and step == 0:
        return {"return": 0.0, "length": max_traj_len}
    return evaluate_robo(
        agent,
        env,
        max_traj_len=max_traj_len,
        num_episodes=num_episodes,
        save_video=save_video,
    )


def _sample_action(agent, observation):
    action, agent = agent.sample_actions(observation)
    return np.asarray(action), agent


def _load_robomimic_dataset(dataset_path):
    seq_dataset = SequenceDataset(
        hdf5_path=str(dataset_path),
        obs_keys=OBS_KEYS,
        dataset_keys=("actions", "rewards", "dones"),
        hdf5_cache_mode="all",
        load_next_obs=True,
    )
    return process_robomimic_dataset(seq_dataset)


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    assert FLAGS.checkpoint_keep > 0, FLAGS.checkpoint_keep
    assert FLAGS.max_iter >= 0, FLAGS.max_iter
    assert FLAGS.trajs_per_update > 0, FLAGS.trajs_per_update
    assert FLAGS.grad_updates_per_iter > 0, FLAGS.grad_updates_per_iter
    assert FLAGS.env_name in ENV_TO_HORIZON_MAP, (
        f"Public release only supports robomimic tasks {sorted(ENV_TO_HORIZON_MAP)}; "
        f"got env_name={FLAGS.env_name!r}"
    )

    code_root = os.path.dirname(os.path.abspath(__file__))
    wandb_init_kwargs = {
        "project": FLAGS.project_name,
        "tags": FLAGS.wandb_tags,
    }
    if FLAGS.wandb_run_group != "":
        wandb_init_kwargs["group"] = FLAGS.wandb_run_group
    if FLAGS.wandb_entity is not None:
        wandb_init_kwargs["entity"] = FLAGS.wandb_entity
    run = wandb.init(**wandb_init_kwargs)
    if FLAGS.wandb_log_code:
        include_fn = _build_source_code_include_fn(code_root)
        exclude_fn = _build_gitignore_exclude_fn(code_root)
        run.log_code(root=code_root, include_fn=include_fn, exclude_fn=exclude_fn)
    wandb_cfg = FLAGS.config.to_dict()
    for k in FLAGS:
        if k == "config" or k.startswith("config."):
            continue
        wandb_cfg[k] = FLAGS[k].value
    wandb.config.update(wandb_cfg)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed)

    exp_name = f"{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}__"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f"id{os.environ['SLURM_JOB_ID']}_"
    exp_name += f"s{FLAGS.seed}"

    log_dir = os.path.join(FLAGS.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "flags.json"), "w") as f:
        out = FLAGS.flag_values_dict()
        if "config" in out:
            out["config"] = FLAGS.config.to_dict()
        json.dump(out, f, indent=2)
        f.write("\n")

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    robomimic_root = robomimic_datasets_root(Path("datasets/robomimic"))
    dataset_path = _resolve_robomimic_dataset_path(robomimic_root, FLAGS.env_name, "ph")
    if FLAGS.dataset_dir not in {"", "mh", "ph"}:
        with open(FLAGS.dataset_dir, "rb") as handle:
            dataset = pickle.load(handle)
        dataset["rewards"] = np.asarray(dataset["rewards"]).squeeze()
        dataset["terminals"] = np.asarray(dataset["terminals"]).squeeze()
    elif FLAGS.dataset_dir == "mh":
        dataset = _load_robomimic_dataset(
            _resolve_robomimic_dataset_path(robomimic_root, FLAGS.env_name, "mh")
        )
    else:
        dataset = _load_robomimic_dataset(dataset_path)

    ds = RoboD4RLDataset(env=None, num_data=FLAGS.num_data, custom_dataset=dataset)
    example_observation = ds.dataset_dict["observations"][0][np.newaxis]
    example_action = ds.dataset_dict["actions"][0][np.newaxis]
    env = get_robomimic_env(str(dataset_path), example_action, FLAGS.env_name)
    eval_env = get_robomimic_env(str(dataset_path), example_action, FLAGS.env_name)
    max_traj_len = ENV_TO_HORIZON_MAP[FLAGS.env_name]

    ds.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    assert model_cls in MODEL_REGISTRY, (
        f"Unsupported model_cls={model_cls!r}. "
        f"Supported model classes: {sorted(MODEL_REGISTRY)}"
    )
    create_fn = MODEL_REGISTRY[model_cls].create
    create_sig = inspect.signature(create_fn)
    if "states" in create_sig.parameters and "states" not in kwargs:
        if "states" in ds.dataset_dict:
            state_input = ds.dataset_dict["states"][0][np.newaxis]
        else:
            state_input = example_observation
        agent = create_fn(
            FLAGS.seed,
            example_observation.squeeze(),
            example_action.squeeze(),
            state_input.squeeze(),
            **kwargs,
        )
    else:
        agent = create_fn(FLAGS.seed, example_observation.squeeze(), example_action.squeeze(), **kwargs)
    print_agent_param_summary(agent)

    replay_buffer = RoboReplayBuffer(example_observation.squeeze(), example_action.squeeze(), FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    start_online_step = 0

    train_logger = CsvLogger(os.path.join(log_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(log_dir, "eval.csv"))

    for i in tqdm.tqdm(range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm, dynamic_ncols=True):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update_offline(batch, FLAGS.utd_ratio, FLAGS.pretrain_q, FLAGS.pretrain_r)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)
                train_logger.log({"event": "offline-training", "metric": k, "value": v}, step=i)

        if i % FLAGS.offline_eval_interval == 0:
            eval_info = maybe_evaluate_robo(
                agent,
                eval_env,
                max_traj_len=max_traj_len,
                num_episodes=FLAGS.eval_episodes,
                step=i,
            )

            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)
                eval_logger.log({"event": "offline-evaluation", "metric": k, "value": v}, step=i)

    observations = env.reset()
    total_collected = 0
    iteration = start_online_step

    print(
        f"Trajectory-based collection: collect {FLAGS.trajs_per_update} complete trajectory(s) "
        f"then run {FLAGS.grad_updates_per_iter} gradient update(s)."
    )

    progress = tqdm.tqdm(
        total=FLAGS.max_iter + 1,
        initial=start_online_step,
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        dynamic_ncols=True,
        leave=False,
    )

    while iteration < FLAGS.max_iter + 1:
        trajs_collected = 0
        trajs_successful = 0
        steps_this_collection = 0

        while trajs_collected < FLAGS.trajs_per_update:
            if total_collected < FLAGS.start_training:
                actions = rng.uniform(-1, 1, size=(example_action.shape[1],))
            else:
                actions, agent = _sample_action(agent, observations)

            next_observations, rewards, dones, infos = env.step(actions)
            infos = {} if infos is None else infos
            mask = 1.0 if (not dones or "TimeLimit.truncated" in infos) else 0.0

            replay_buffer.insert(
                dict(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    masks=mask,
                    dones=dones,
                    next_observations=next_observations,
                )
            )

            observations = next_observations
            total_collected += 1
            steps_this_collection += 1

            if dones:
                trajs_collected += 1
                wandb_step = FLAGS.pretrain_steps + total_collected
                if infos.get("success", False) or infos.get("is_success", False):
                    trajs_successful += 1
                if "episode" in infos:
                    for k, v in infos["episode"].items():
                        wandb.log({f"training/env/{k}": v}, step=wandb_step)
                        train_logger.log({"event": "episode", "metric": k, "value": v}, step=wandb_step)
                observations = env.reset()
        traj_success_rate = trajs_successful / trajs_collected

        online_batch_size = int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
        offline_batch_size = int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)

        if total_collected >= FLAGS.start_training and len(replay_buffer) >= online_batch_size:
            for _ in range(FLAGS.grad_updates_per_iter):
                online_batch = replay_buffer.sample(online_batch_size)
                offline_batch = ds.sample(offline_batch_size)

                if FLAGS.offline_ratio == 0.5:
                    batch = combine_half(offline_batch, online_batch, rng)
                else:
                    batch = combine(offline_batch, online_batch, rng)

                if "antmaze" in FLAGS.env_name:
                    batch["rewards"] -= 1

                agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if iteration % FLAGS.log_interval == 0:
                wandb_step = FLAGS.pretrain_steps + total_collected

                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=wandb_step)
                    train_logger.log({"event": "training", "metric": k, "value": v}, step=wandb_step)

                training_metrics = {
                    "training/total_env_steps": total_collected,
                    "training/steps_this_collection": steps_this_collection,
                    "training/trajs_collected": trajs_collected,
                    "training/trajs_successful": trajs_successful,
                    "training/trajs_success_rate": traj_success_rate,
                    "training/iteration": iteration,
                    "training/env/traj_success_rate": traj_success_rate,
                }
                wandb.log(training_metrics, step=wandb_step)
                for k, v in training_metrics.items():
                    train_logger.log({"event": "training", "metric": k, "value": v}, step=wandb_step)

        if iteration % FLAGS.eval_interval == 0:
            wandb_step = FLAGS.pretrain_steps + total_collected
            save_video_this_eval = FLAGS.save_video and (iteration % (FLAGS.eval_interval * 2) == 0)
            eval_info = evaluate_robo(
                agent,
                eval_env,
                max_traj_len=max_traj_len,
                num_episodes=FLAGS.eval_episodes,
                save_video=save_video_this_eval,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=wandb_step)
                eval_logger.log({"event": "evaluation", "metric": k, "value": v}, step=wandb_step)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(chkpt_dir, agent, step=iteration, keep=FLAGS.checkpoint_keep, overwrite=True)
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, "buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")

        iteration += 1
        progress.update(1)

    progress.close()

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    sys.argv = _dedupe_config_overrides(sys.argv)
    app.run(main, argv=sys.argv)
