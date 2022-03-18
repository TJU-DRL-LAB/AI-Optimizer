from typing import cast

import gym
from gym.spaces import Discrete


def get_action_size_from_env(env: gym.Env) -> int:
    if isinstance(env.action_space, Discrete):
        return cast(int, env.action_space.n)
    return cast(int, env.action_space.shape[0])
