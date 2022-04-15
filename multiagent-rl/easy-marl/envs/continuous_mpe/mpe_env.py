from gym.spaces import Box, Dict
import numpy as np

from envs.continuous_mpe.multiagent import MultiAgentEnv
import envs.continuous_mpe.multiagent.scenarios as scenarios

from envs.base_env import BaseEnvironment


class MPEEnv(BaseEnvironment):
    def __init__(self, args):
        self.args = args

        # register mpe env
        scenario = scenarios.load(self.args.scenario_name + ".py").Scenario()
        world = scenario.make_world()
        self.env = MultiAgentEnv(world, 
                                 scenario.reset_world, 
                                 scenario.reward,
                                 scenario.observation,
                                 discrete_action=False)
        self.env.discrete_action_input = False
        self.agents = self.env.agents

        # get continuous_mpe info
        self.agent_count = self.env.n
        self.episode_max_step = args.episode_max_step

        # rl obs & action space
        self.state_dim = sum([obs_space.shape[0] for obs_space in self.env.observation_space])
        self.observation_space = [obs_space for obs_space in self.env.observation_space]
        self.state_space = Box(low=-float('inf'), high=float('inf'), shape=(self.state_dim,))
        self.action_space = self.env.action_space
        self.is_discrete = False

        self.step_count = 0

    def reset(self):
        self.env._reset()
        self.step_count = 0

        observation_list = self._get_obs()
        state = self._get_state()
        return (observation_list, state)

    def step(self, action_list):
        # continuous mpe inner simulate
        _, reward, done, info = self.env._step(action_list)

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
        return [self.env._get_obs(agent) for agent in self.agents]

    def _get_state(self):
        return np.concatenate([self.env._get_obs(agent) for agent in self.agents], axis=0)

    def _get_reward(self, reward):
        return ([self.env._get_reward(agent) for agent in self.agents], sum([self.env._get_reward(agent) for agent in self.agents]))

    def _get_done(self, done):
        if self.step_count >= self.episode_max_step:
            return True
        return False

    def _get_info(self, info):
        infos = {
            "step": self.step_count,
        }
        return infos
