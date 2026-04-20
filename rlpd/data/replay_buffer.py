import collections
from typing import Dict, Optional, Union

import gym
import gym.spaces
import jax
import numpy as np
from flax.core import frozen_dict

from rlpd.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = _coerce_replay_value(dataset_dict, data_dict)
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _coerce_replay_value(dataset_array: np.ndarray, value):
    value_array = np.asarray(value)
    if dataset_array.dtype != np.uint8 or not np.issubdtype(value_array.dtype, np.floating):
        return value
    if dataset_array.ndim < 4 or dataset_array.shape[-1] <= 0 or dataset_array.shape[-1] % 3 != 0:
        return value
    lo = float(np.min(value_array))
    hi = float(np.max(value_array))
    if not (0.0 <= lo and hi <= 255.0):
        raise ValueError(f"Expected float images in [0,1] or [0,255], got range [{lo}, {hi}]")
    if hi <= 1.0:
        return (255.0 * value_array).astype(np.uint8)
    return value_array.astype(np.uint8)


def _insert_dataset_arrays(dataset_dict: DatasetDict, dataset: Dict[str, np.ndarray], capacity: int):
    dataset_size = len(dataset["observations"])
    if dataset_size <= capacity:
        for key in dataset_dict:
            if key in dataset:
                dataset_dict[key][:dataset_size] = _coerce_replay_value(dataset_dict[key], dataset[key])
        size = dataset_size
    else:
        indices = np.random.choice(dataset_size, capacity, replace=False)
        for key in dataset_dict:
            if key in dataset:
                dataset_dict[key][:] = _coerce_replay_value(dataset_dict[key], dataset[key][indices])
        size = capacity
    return size, size % capacity


def _init_robo_replay_dict(example_observation: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    # if isinstance(obs_space, gym.spaces.Box):
    return np.empty((capacity, *example_observation.shape), dtype=example_observation.dtype)


def _device_put_numeric_leaves(batch):
    was_frozen = isinstance(batch, frozen_dict.FrozenDict)
    if was_frozen:
        batch = batch.unfreeze()

    def convert(value):
        if not isinstance(value, np.ndarray):
            return value
        if value.dtype.kind in {"O", "S", "U"}:
            return value
        return jax.device_put(value)

    batch = jax.tree_util.tree_map(convert, batch)
    if was_frozen:
        return frozen_dict.freeze(batch)
    return batch


class RoboReplayBuffer(Dataset):
    def __init__(
        self,
        example_observation,
        example_action,
        capacity: int,
    ):
        observation_data = _init_robo_replay_dict(example_observation, capacity)
        next_observation_data = _init_robo_replay_dict(example_observation, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *example_action.shape), dtype=example_action.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_batch(self, data_dict: Dict[str, np.ndarray]):
        """Insert a batch of transitions at once. More efficient than repeated insert() calls."""
        batch_size = len(data_dict["observations"])
        if batch_size == 0:
            return

        # Calculate indices for insertion (handling wrap-around)
        indices = np.arange(self._insert_index, self._insert_index + batch_size) % self._capacity

        for key in self.dataset_dict.keys():
            if key in data_dict:
                self.dataset_dict[key][indices] = data_dict[key]

        self._insert_index = (self._insert_index + batch_size) % self._capacity
        self._size = min(self._size + batch_size, self._capacity)

    def insert_dataset(self, dataset: Dict[str, np.ndarray]):
        self._size, self._insert_index = _insert_dataset_arrays(self.dataset_dict, dataset, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(_device_put_numeric_leaves(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_dataset(self, dataset: Dict[str, np.ndarray]):
        self._size, self._insert_index = _insert_dataset_arrays(self.dataset_dict, dataset, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(_device_put_numeric_leaves(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
