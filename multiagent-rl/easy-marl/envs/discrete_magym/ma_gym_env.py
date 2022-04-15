import gym
from gym.spaces import Box, Dict, Tuple
from envs.discrete_magym.register_env import register_ma_gym_envs
from envs.base_env import BaseEnvironment
import numpy as np


class MAGYMEnv(BaseEnvironment):
    def __init__(self, args):
        self.args = args

        # register ma-gym env
        register_ma_gym_envs()
        self.env = gym.make(self.args.scenario_name)

        # get ma-gym info
        self.agent_count = self.env.n_agents

        # rl obs & action space
        self.state_dim = sum([obs_space.shape[0] for obs_space in self.env.observation_space])
        self.observation_space = [obs_shape for obs_shape in self.env.observation_space]
        self.state_space = Box(low=-float('inf'), high=float('inf'), shape=(self.state_dim,))
        self.action_space = [act_space for act_space in self.env.action_space]
        self.is_discrete = True

        self.step_count = 0

    def reset(self):
        self.env.reset()
        self.step_count = 0

        observation_list = self._get_obs()
        state = self._get_state()
        return (observation_list, state)

    def step(self, action_list):
        # discrete ma-gym inner simulate
        _, reward, done, info = self.env.step(action_list)

        # get rl info
        observation_list = self._get_obs()
        state = self._get_state()
        reward_list, team_reward = self._get_reward(reward)
        done = self._get_done(done)
        info = self._get_info(info)

        self.step_count += 1
        return (observation_list, state), (reward_list, team_reward), done, info

    def render(self):
        self.env.render()

    def _get_obs(self):
        return [np.array(obs) for obs in self.env.get_agent_obs()]

    def _get_state(self):
        return np.concatenate(self.env.get_agent_obs(), axis=0)

    def _get_reward(self, reward):
        return (reward, sum(reward))

    def _get_done(self, done):
        if [True for _ in range(len(done))] == done:
            return True
        return False

    def _get_info(self, info):
        infos = {
            "step": self.step_count,
        }
        return infos
