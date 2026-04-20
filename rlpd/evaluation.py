from typing import Dict

import gym
import numpy as np


class SamplerPolicy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, observations, deterministic=False, add_noise=0.0, **kwargs):
        actions = self.agent.eval_actions(observations)
        if isinstance(actions, tuple) and len(actions) == 2:
            actions, self.agent = actions
        return np.asarray(actions)


class TrajSampler:
    def __init__(self, env, max_traj_length=1000):
        self._env = env
        self.max_traj_length = max_traj_length

    def sample(self, policy, n_trajs, deterministic=False, add_noise=0.0, filter=False):
        trajs = []
        for _ in range(n_trajs):
            observation = self._env.reset()
            rewards = []
            steps = []
            traj_steps = 0
            done = False

            while not done and traj_steps < self.max_traj_length:
                action = np.asarray(
                    policy(
                        observation,
                        deterministic=deterministic,
                        add_noise=add_noise,
                    )
                )
                observation, reward, done, info = self._env.step(action)
                info = {} if info is None else info
                rewards.append(reward)
                step_count = int(info.get("chunk_steps", 1))
                steps.append(step_count)
                traj_steps += step_count

            if filter and not np.sum(rewards) > 0:
                continue

            trajs.append(
                {
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "steps": np.asarray(steps, dtype=np.int32),
                }
            )

        return trajs

    @property
    def env(self):
        return self._env


def evaluate_robo(
    agent,
    env: gym.Env,
    num_episodes: int,
    max_traj_len: int,
    save_video: bool = False,
    return_trajs: bool = False,
) -> Dict[str, float]:
    sampler = TrajSampler(env, max_traj_len)
    policy = SamplerPolicy(agent)
    trajs = sampler.sample(policy, num_episodes)
    lengths = [np.sum(t["steps"]) for t in trajs]
    returns = [np.sum(t["rewards"]) for t in trajs]
    metrics = {"return": float(np.mean(returns)), "length": float(np.mean(lengths))}
    if return_trajs:
        return trajs, metrics
    return metrics
