from gym.spaces import Box, Discrete
import numpy as np

class ParticleEntity(object):
    def __init__(self):
        self.position = None


class Environment(object):
    """
    task description:
    all agents try to meet at the same target_position of a 2D-plane
    """
    def __init__(self):
        self.agent_count = 2
        self.env_bound = 4
        self.env_dim = 2  # do not change this!
        self.action_movement_map = {0: np.array([0, 0]),  # stop
                                    1: np.array([0, 1]),  # up
                                    2: np.array([1, 0]),  # right
                                    3: np.array([0, -1]),  # down
                                    4: np.array([-1, 0])}  # left
        self.action_effective_step = 1
        self.sparse_reward_flag = False
        self.agents = [ParticleEntity() for _ in range(self.agent_count)]
        self.target_position = None
        self.step_count = 0
        self.max_step_count = 40

        self.observation_space = [Box(low=-float('inf'), high=float('inf'), shape=(4,)) for _ in range(self.agent_count)]
        self.state_space = Box(low=-float('inf'), high=float('inf'), shape=(8,))
        self.action_space = [Discrete(5) for _ in range(self.agent_count)]
        self.is_discrete = True

    def reset(self):
        for i, agent in enumerate(self.agents):
            agent.position = np.random.randint(0, self.env_bound, self.env_dim)  # discrete space [low, high)
        self.target_position = np.random.randint(0, self.env_bound, self.env_dim)

        observation_list = self._get_observation_list()
        state = self._get_state()
        return observation_list, state

    def step(self, action_list):
        for _ in range(self.action_effective_step):
            self._simulate_one_step(action_list)

        observation_list = self._get_observation_list()
        state = self._get_state()
        reward_list, team_reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return (observation_list, state), (reward_list, team_reward), done, info

    def render(self):
        # TODO
        raise NotImplementedError()

    def _get_observation_list(self):
        return [self._get_observation(agent) for agent in self.agents]

    def _get_observation(self, agent):
        relative_position = []
        for other in self.agents:
            if other is agent:
                continue
            relative_position.append(other.position - agent.position)
        observation = np.concatenate([agent.position] + relative_position)  # each observation has a shape of (-1, )
        return observation * 1.0 / self.env_bound  # always normalize the observation

    def _get_state(self):
        """
        concat all agent's observation to construct state info
        """
        return np.concatenate([self._get_observation(agent) for agent in self.agents], axis=0)  # state shape is (-1, )

    def _simulate_one_step(self, action_list):
        self.step_count += 1
        for i, agent in enumerate(self.agents):
            agent_movement = self.action_movement_map[action_list[i]]
            agent.position = agent.position + agent_movement
            agent.position = agent.position % self.env_bound  # keep the agents in the env_bound

    def _get_reward(self):
        if self.sparse_reward_flag:
            reward_list = []
            for agent in self.agents:
                if (agent.position != self.target_position).any():
                    reward_list.append(0)
                else:
                    reward_list.append(1)
        else:
            reward_list = []
            for agent in self.agents:
                distance = np.sqrt(np.sum(np.square(agent.position - self.target_position)))
                reward_list.append(-distance)  # negative distance
        reward_list = reward_list + [sum(reward_list)]
        # reward_list[-1] is the global reward (i.e., sum of individual rewards)
        return (reward_list, sum(reward_list))

    def _get_done(self):
        if self.step_count >= self.max_step_count:
            return True
        return False

    def _get_info(self):
        info = {
            'step_count': self.step_count,
            'max_step_count': self.max_step_count
        }
        return info

