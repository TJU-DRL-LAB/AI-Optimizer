
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics import FetchPushEnv as FetchPushEnv
from . import register_env
import time

@register_env('fetch_push')
class MultiFetchPushEnv(FetchPushEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        FetchPushEnv.__init__(self, reward_type='dense')
        self.initial_object_xpos = self.sim.data.get_site_xpos('object0')
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        goals = self.initial_gripper_xpos[:3] + np.random.uniform(-0.15, 0.15, size=(num_tasks,3))
        #goals += self.target_offset
        #print("goals:", goals)
        goals[:,2] = self.height_offset
        #print("height_offset:",self.height_offset)
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            #print("goal2+")
            goals[:,2] += self.np_random.uniform(0, 0.45, size=2)
        #positions = np.random.uniform(0.4, 0.5, size=(num_tasks,))
        #print("goal_new:", goals)

        tasks = [{'goal': d} for d in goals]
        return tasks
    def step(self, action):
        #print(self.initial_object_xpos)
        #print(self.initial_gripper_xpos[:3])
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
        success = self._is_success(obs['achieved_goal'], self._goal)
        reward = self.compute_reward(obs['achieved_goal'], self._goal, info)
        #self.has_object = False
        #obs_temp = self._get_obs()
        done_g_l = obs_[:3]
        #print(self.initial_gripper_xpos[:3])
        #self.has_object = True
        return obs_, reward, done, dict(reward_dense=reward, done_g=obs['achieved_goal'], done_g_l=done_g_l, success=success)



