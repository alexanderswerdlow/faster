from copy import deepcopy

import numpy as np
from robomimic.config import config_factory
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from rlpd.data.dataset import Dataset


OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
ENV_TO_HORIZON_MAP = {
    "lift": 400,
    "can": 400,
    "square": 400,
    "transport": 700,
    "tool_hang": 700,
}


def _patch_robosuite_offscreen_context():
    import robosuite.utils.binding_utils as binding_utils

    render_cls = binding_utils.MjRenderContext
    if getattr(render_cls, "_sample_rank_make_current_patch", False):
        return

    original_render = render_cls.render
    original_read_pixels = render_cls.read_pixels

    def patched_render(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original_render(self, *args, **kwargs)

    def patched_read_pixels(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original_read_pixels(self, *args, **kwargs)

    render_cls.render = patched_render
    render_cls.read_pixels = patched_read_pixels
    render_cls._sample_rank_make_current_patch = True


def _load_robomimic_env_meta(dataset_path):
    env_meta = deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    assert "env_kwargs" in env_meta, sorted(env_meta)
    env_meta["env_kwargs"]["hard_reset"] = False
    return env_meta


def _reset_robomimic_playback_env(env):
    if hasattr(env, "env") and hasattr(env.env, "hard_reset"):
        env.env.hard_reset = False
    env.reset()
    state_dict = env.get_state()
    assert "states" in state_dict, sorted(state_dict)
    return env.reset_to({"states": state_dict["states"]})


class RobosuiteGymWrapper:
    def __init__(self, env, horizon, example_action):
        self.env = env
        self.horizon = horizon
        self.action_space = example_action
        self.timestep = 0
        self.returns = 0.0

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        next_obs = self._process_obs(next_obs)
        success = self.env.is_success()["task"]
        self.timestep += 1
        self.returns += reward
        timeout = self.timestep >= self.horizon
        terminated = done or success or timeout
        info = None
        if terminated:
            info = {"episode": {"return": self.returns, "length": self.timestep}}
            if timeout and not success:
                info["TimeLimit.truncated"] = True
        return next_obs, reward, terminated, info

    def reset(self):
        obs = _reset_robomimic_playback_env(self.env)
        obs = self._process_obs(obs)
        self.timestep = 0
        self.returns = 0.0
        return obs

    def render(self, mode, height=None, width=None):
        return self.env.render(mode=mode, height=height, width=width)

    def _process_obs(self, obs):
        return np.concatenate([obs[key] for key in OBS_KEYS], axis=-1)


def process_robomimic_dataset(seq_dataset):
    cached = seq_dataset.getitem_cache
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    for item in cached:
        observations.append(np.concatenate([item["obs"][key] for key in OBS_KEYS], axis=1))
        next_observations.append(
            np.concatenate([item["next_obs"][key] for key in OBS_KEYS], axis=1)
        )
        actions.append(np.asarray(item["actions"]))
        rewards.append(np.asarray(item["rewards"]))
        terminals.append(np.asarray(item["dones"]))

    return {
        "observations": np.concatenate(observations).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "rewards": np.concatenate(rewards).astype(np.float32),
        "terminals": np.concatenate(terminals).astype(np.float32),
        "next_observations": np.concatenate(next_observations).astype(np.float32),
    }


def get_robomimic_env(dataset_path, example_action, env_name):
    assert env_name in ENV_TO_HORIZON_MAP, env_name
    _patch_robosuite_offscreen_context()
    ObsUtils.initialize_obs_utils_with_config(config_factory(algo_name="iql"))
    env_meta = _load_robomimic_env_meta(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return RobosuiteGymWrapper(env, ENV_TO_HORIZON_MAP[env_name], example_action)


def _episode_dones(observations, next_observations, terminals, ignore_done):
    dones = np.zeros_like(terminals, dtype=np.float32)
    for i in range(len(dones) - 1):
        transition_break = (
            np.linalg.norm(observations[i + 1] - next_observations[i]) > 1e-6
        )
        if ignore_done:
            dones[i] = float(transition_break)
        else:
            dones[i] = float(transition_break or terminals[i] == 1.0)
    dones[-1] = 1.0
    return dones


def _truncate_dataset_by_episodes(dataset_dict, num_data):
    done_indices = [-1] + [i for i, done in enumerate(dataset_dict["dones"]) if done]
    keep = []
    for i in range(len(done_indices) - 1):
        if done_indices[i] + 1 < done_indices[i + 1]:
            keep.append(done_indices[i])
    keep.append(done_indices[-1])
    total_len = keep[num_data] - keep[0]
    for key, value in dataset_dict.items():
        dataset_dict[key] = value[:total_len]


class RoboD4RLDataset(Dataset):
    def __init__(
        self,
        env,
        clip_to_eps=True,
        eps=1e-5,
        num_data=0,
        ignore_done=False,
        custom_dataset=None,
    ):
        assert custom_dataset is not None, (
            "Public release RoboD4RLDataset only supports custom_dataset input."
        )
        dataset = {key: np.asarray(value).copy() for key, value in custom_dataset.items()}
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones = _episode_dones(
            dataset["observations"],
            dataset["next_observations"],
            dataset["terminals"],
            ignore_done,
        )
        dataset_dict = {
            "observations": dataset["observations"].astype(np.float32),
            "actions": dataset["actions"].astype(np.float32),
            "rewards": dataset["rewards"].astype(np.float32),
            "masks": 1.0 - dataset["terminals"].astype(np.float32),
            "dones": dones.astype(np.float32),
            "next_observations": dataset["next_observations"].astype(np.float32),
        }
        if num_data != 0:
            _truncate_dataset_by_episodes(dataset_dict, num_data)
        super().__init__(dataset_dict)
