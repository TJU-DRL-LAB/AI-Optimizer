import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_border
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class PongDuel(gym.Env):
    """Two Player Pong Game - Competitive"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, step_cost=0, reward=1, max_rounds=10):
        self._grid_shape = (40, 30)
        self.n_agents = 2
        self.reward = reward
        self._max_rounds = max_rounds
        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(self.n_agents)])

        self._step_count = None
        self._step_cost = step_cost
        self._total_episode_reward = None
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self._agent_dones = None
        self.ball_pos = None
        self.__rounds = None

        # agent pos(2), ball pos (2), balldir (6-onehot)
        self._obs_low = np.array([0., 0., 0., 0.] + [0.] * len(BALL_DIRECTIONS), dtype=np.float32)
        self._obs_high = np.array([1., 1., 1., 1.] + [1.] * len(BALL_DIRECTIONS), dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)for _ in range(self.n_agents)])

        self.curr_ball_dir = None
        self.viewer = None
        self.seed()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __update_agent_view(self, agent_i):
        for row in range(self.agent_prev_pos[agent_i][0] - PADDLE_SIZE,
                         self.agent_prev_pos[agent_i][0] + PADDLE_SIZE + 1):
            self._full_obs[row][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']

        for row in range(self.agent_pos[agent_i][0] - PADDLE_SIZE, self.agent_pos[agent_i][0] + PADDLE_SIZE + 1):
            self._full_obs[row][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1) \
                                                              + '_' + str(row - self.agent_pos[agent_i][0])

    def __update_ball_view(self):
        self._full_obs[self.ball_pos[0]][self.ball_pos[1]] = PRE_IDS['ball']

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1],
                                   cell_size=CELL_SIZE, fill='white', line_color='white')

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()
        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)

        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)

        self.__update_ball_view()

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []

        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]]

            pos = self.ball_pos
            _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]]

            _ball_dir = [0 for _ in range(len(BALL_DIRECTIONS))]
            _ball_dir[BALL_DIRECTIONS.index(self.curr_ball_dir)] = 1

            _agent_i_obs += _ball_dir  # one hot ball dir encoding

            _obs.append(_agent_i_obs)

        return _obs

    def __init_ball_pos(self):
        self.ball_pos = [self.np_random.randint(5, self._grid_shape[0] - 5), self.np_random.randint(10, self._grid_shape[1] - 10)]
        self.curr_ball_dir = self.np_random.choice(['NW', 'SW', 'SE', 'NE'])

    def reset(self):
        self.__rounds = 0
        self.agent_pos[0] = (self.np_random.randint(PADDLE_SIZE, self._grid_shape[0] - PADDLE_SIZE - 1), 1)
        self.agent_pos[1] = (self.np_random.randint(PADDLE_SIZE, self._grid_shape[0] - PADDLE_SIZE - 1),
                             self._grid_shape[1] - 2)
        self.agent_prev_pos = {_: self.agent_pos[_] for _ in range(self.n_agents)}
        self.__init_ball_pos()
        self._agent_dones = [False, False]
        self.__init_full_obs()
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]

        return self.get_agent_obs()

    @property
    def __ball_cells(self):
        if self.curr_ball_dir == 'E':
            return [self.ball_pos, [self.ball_pos[0], self.ball_pos[1] - 1], [self.ball_pos[0], self.ball_pos[1] - 2]]
        if self.curr_ball_dir == 'W':
            return [self.ball_pos, [self.ball_pos[0], self.ball_pos[1] + 1], [self.ball_pos[0], self.ball_pos[1] + 2]]
        if self.curr_ball_dir == 'NE':
            return [self.ball_pos, [self.ball_pos[0] + 1, self.ball_pos[1] - 1],
                    [self.ball_pos[0] + 2, self.ball_pos[1] - 2]]
        if self.curr_ball_dir == 'NW':
            return [self.ball_pos, [self.ball_pos[0] + 1, self.ball_pos[1] + 1],
                    [self.ball_pos[0] + 2, self.ball_pos[1] + 2]]
        if self.curr_ball_dir == 'SE':
            return [self.ball_pos, [self.ball_pos[0] - 1, self.ball_pos[1] - 1],
                    [self.ball_pos[0] - 2, self.ball_pos[1] - 2]]
        if self.curr_ball_dir == 'SW':
            return [self.ball_pos, [self.ball_pos[0] - 1, self.ball_pos[1] + 1],
                    [self.ball_pos[0] - 2, self.ball_pos[1] + 2]]

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for row in range(self.agent_pos[agent_i][0] - 2, self.agent_pos[agent_i][0] + 3):
                fill_cell(img, (row, self.agent_pos[agent_i][1]), cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])

        ball_cells = self.__ball_cells
        fill_cell(img, ball_cells[0], cell_size=CELL_SIZE, fill=BALL_HEAD_COLOR)
        fill_cell(img, ball_cells[1], cell_size=CELL_SIZE, fill=BALL_TAIL_COLOR)
        fill_cell(img, ball_cells[2], cell_size=CELL_SIZE, fill=BALL_TAIL_COLOR)

        img = draw_border(img, border_width=2, fill='gray')

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        if move == 0:  # noop
            next_pos = None
        elif move == 1:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 2:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and PADDLE_SIZE <= next_pos[0] <= (self._grid_shape[0] - PADDLE_SIZE - 1):
            self.agent_prev_pos[agent_i] = self.agent_pos[agent_i]
            self.agent_pos[agent_i] = next_pos
            self.__update_agent_view(agent_i)

    def __update_ball_pos(self):

        if self.ball_pos[0] <= 1:
            self.curr_ball_dir = 'SE' if self.curr_ball_dir == 'NE' else 'SW'
        elif self.ball_pos[0] >= (self._grid_shape[0] - 2):
            self.curr_ball_dir = 'NE' if self.curr_ball_dir == 'SE' else 'NW'
        elif PRE_IDS['agent'] in self._full_obs[self.ball_pos[0]][self.ball_pos[1] + 1]:
            edge = int(self._full_obs[self.ball_pos[0]][self.ball_pos[1] + 1].split('_')[1])
            _dir = ['NW', 'W', 'SW']
            if edge <= 0:
                _p = [0.25 + ((1 - 0.25) / PADDLE_SIZE * (abs(edge))),
                      0.5 - (0.5 / PADDLE_SIZE * (abs(edge))),
                      0.25 - (0.25 / PADDLE_SIZE * (abs(edge))), ]
            elif edge >= 0:
                _p = [0.25 - (0.25 / PADDLE_SIZE * (abs(edge))),
                      0.5 - (0.5 / PADDLE_SIZE * (abs(edge))),
                      0.25 + ((1 - 0.25) / PADDLE_SIZE * (abs(edge)))]
            _p[len(_dir) // 2] += 1 - sum(_p)

            self.curr_ball_dir = self.np_random.choice(_dir, p=_p)
        elif PRE_IDS['agent'] in self._full_obs[self.ball_pos[0]][self.ball_pos[1] - 1]:
            _dir = ['NE', 'E', 'SE']
            edge = int(self._full_obs[self.ball_pos[0]][self.ball_pos[1] - 1].split('_')[1])
            if edge <= 0:
                _p = [0.25 + ((1 - 0.25) / PADDLE_SIZE * (abs(edge))),
                      0.5 - (0.5 / PADDLE_SIZE * (abs(edge))),
                      0.25 - (0.25 / PADDLE_SIZE * (abs(edge))), ]
            elif edge >= 0:
                _p = [0.25 - (0.25 / PADDLE_SIZE * (abs(edge))),
                      0.5 - (0.5 / PADDLE_SIZE * (abs(edge))),
                      0.25 + ((1 - 0.25) / PADDLE_SIZE * (abs(edge)))]
            _p[len(_dir) // 2] += 1 - sum(_p)
            self.curr_ball_dir = self.np_random.choice(_dir, p=_p)

        if self.curr_ball_dir == 'E':
            new_ball_pos = self.ball_pos[0], self.ball_pos[1] + 1
        elif self.curr_ball_dir == 'W':
            new_ball_pos = self.ball_pos[0], self.ball_pos[1] - 1
        elif self.curr_ball_dir == 'NE':
            new_ball_pos = self.ball_pos[0] - 1, self.ball_pos[1] + 1
        elif self.curr_ball_dir == 'NW':
            new_ball_pos = self.ball_pos[0] - 1, self.ball_pos[1] - 1
        elif self.curr_ball_dir == 'SE':
            new_ball_pos = self.ball_pos[0] + 1, self.ball_pos[1] + 1
        elif self.curr_ball_dir == 'SW':
            new_ball_pos = self.ball_pos[0] + 1, self.ball_pos[1] - 1

        self.ball_pos = new_ball_pos

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def step(self, action_n):
        assert len(action_n) == self.n_agents
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # if ball is beyond paddle, initiate a new round
        if self.ball_pos[1] < 1:
            rewards = [0, self.reward]
            self.__rounds += 1
        elif self.ball_pos[1] >= (self._grid_shape[1] - 1):
            rewards = [self.reward, 0]
            self.__rounds += 1

        if self.__rounds == self._max_rounds:
            self._agent_dones = [True for _ in range(self.n_agents)]
        else:
            for agent_i in range(self.n_agents):
                self.__update_agent_pos(agent_i, action_n[agent_i])

            if (self.ball_pos[1] < 1) or (self.ball_pos[1] >= self._grid_shape[1] - 1):
                self.__init_ball_pos()
            else:
                self.__update_ball_pos()

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'rounds': self.__rounds}


CELL_SIZE = 5

ACTION_MEANING = {
    0: "NOOP",
    1: "UP",
    2: "DOWN",
}

AGENT_COLORS = {
    0: 'red',
    1: 'blue'
}
WALL_COLOR = 'black'
BALL_HEAD_COLOR = 'orange'
BALL_TAIL_COLOR = 'yellow'

# each pre-id should be unique and single char
PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'ball': 'B',
    'empty': 'O'
}

BALL_DIRECTIONS = ['NW', 'W', 'SW', 'SE', 'E', 'NE']
PADDLE_SIZE = 2
