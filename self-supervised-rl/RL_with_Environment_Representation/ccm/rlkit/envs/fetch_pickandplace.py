
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics import FetchPickAndPlaceEnv as FetchPickAndPlaceEnv
from . import register_env

@register_env('fetch_pickplace')
class MultiFetchpickplaceEnv(FetchPickAndPlaceEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        FetchPickAndPlaceEnv.__init__(self, reward_type='dense')
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        goals = [0.405, 0.48, 0.0] + np.random.uniform(-0.15, 0.15, size=(num_tasks,3))
        #goals += self.target_offset
        goals[:,2] = self.height_offset
        if self.target_in_the_air and np.random.uniform() < 0.5:
            goal[:,2] += np.random.uniform(0, 0.45, size=2)
        #positions = np.random.uniform(0.4, 0.5, size=(num_tasks,))
        tasks = [{'goal': d} for d in goals]
        return tasks
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        obs['desired_goal'] = self._goal.copy()
        obs_ = obs['observation']
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self._goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self._goal, info)
        return obs_, reward, done, dict(reward_dense=reward, done_g=obs['achieved_goal'])



