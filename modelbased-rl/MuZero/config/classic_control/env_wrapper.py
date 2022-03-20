from collections import deque

import numpy as np

from core.game import Game, Action


class ClassicControlWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount)
        self.k = k
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [Action(_) for _ in range(self.env.action_space.n)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.obs(len(self.rewards)), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):
            self.obs_history.append(obs)

        return self.obs(0)

    def obs(self, i):
        frames = self.obs_history[i:i + self.k]
        return np.array(frames).flatten()

    def close(self):
        self.env.close()
