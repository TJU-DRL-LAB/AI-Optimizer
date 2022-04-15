import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_cell_outline, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class Switch(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable: bool = False, step_cost: float = 0, n_agents: int = 4, max_steps: int = 50,
                 clock: bool = True):
        assert 2 <= n_agents <= 4, 'Number of Agents has to be in range [2,4]'
        self._grid_shape = (3, 7)
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self._total_episode_reward = None
        self._add_clock = clock

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])  # l,r,t,d,noop

        init_agent_pos = {0: [0, 1], 1: [0, self._grid_shape[1] - 2],
                          2: [2, 1], 3: [2, self._grid_shape[1] - 2]}
        final_agent_pos = {0: [0, self._grid_shape[1] - 1], 1: [0, 0],
                           2: [2, self._grid_shape[1] - 1], 3: [2, 0]}  # they have to go in opposite direction

        self.init_agent_pos, self.final_agent_pos = {}, {}
        for agent_i in range(n_agents):
            self.init_agent_pos[agent_i] = init_agent_pos[agent_i]
            self.final_agent_pos[agent_i] = final_agent_pos[agent_i]

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self.__init_full_obs()
        self.viewer = None

        self.full_observable = full_observable
        # agent pos (2)
        self._obs_high = np.ones(2 + (1 if self._add_clock else 0))
        self._obs_low = np.zeros(2 + (1 if self._add_clock else 0))
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in range(self.n_agents)])
        self.seed()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR)

        for agent_i, pos in list(self.final_agent_pos.items())[:self.n_agents]:
            row, col = pos[0], pos[1]
            draw_cell_outline(self._base_img, (row, col), cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])

    def __create_grid(self):
        _grid = -1 * np.ones(self._grid_shape)  # all are walls
        _grid[self._grid_shape[0] // 2, :] = 0  # road in the middle
        _grid[:, [0, 1]] = 0
        _grid[:, [-1, -2]] = 0
        return _grid

    def __init_full_obs(self):
        self.agent_pos = copy.copy(self.init_agent_pos)
        self._full_obs = self.__create_grid()
        for agent_i, pos in self.agent_pos.items():
            self.__update_agent_view(agent_i)
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(0, self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [round(pos[0] / (self._grid_shape[0] - 1), 2),
                            round(pos[1] / (self._grid_shape[1] - 1), 2)]
            if self._add_clock:
                _agent_i_obs += [self._step_count / self._max_steps]  # add current step count (for time reference)
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]

        return _obs

    def reset(self):
        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return self._base_grid[row, col] == -1

    def _is_cell_vacant(self, pos):
        is_valid = (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])
        return is_valid and (self._full_obs[pos[0], pos[1]] == 0)

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0], curr_pos[1]] = 0
            self.__update_agent_view(agent_i)
        else:
            pass

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0], self.agent_pos[agent_i][1]] = agent_i + 1

    def __is_agent_done(self, agent_i):
        return self.agent_pos[agent_i] == self.final_agent_pos[agent_i]

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]
        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

                self._agent_dones[agent_i] = self.__is_agent_done(agent_i)
                if self._agent_dones[agent_i]:
                    rewards[agent_i] = 5
            else:
                rewards[agent_i] = 0

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i], radius=0.3)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange'
}

CELL_SIZE = 30

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}
