# -*- coding: utf-8 -*-

import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class Combat(gym.Env):
    """
    We simulate a simple battle involving two opposing teams in a n x n grid.
    Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
    square around the team center, which is picked uniformly in the grid. At each time step, an agent can
    perform one of the following actions: move one cell in one of four directions; attack another agent
    by specifying its ID j (there are m attack actions, each corresponding to one enemy agent); or do
    nothing. If agent A attacks agent B, then B’s health point will be reduced by 1, but only if B is inside
    the firing range of A (its surrounding 3 × 3 area). Agents need one time step of cooling down after
    an attack, during which they cannot attack. All agents start with 3 health points, and die when their
    health reaches 0. A team will win if all agents in the other team die. The simulation ends when one
    team wins, or neither of teams win within 40 time steps (a draw).

    The model controls one team during training, and the other team consist of bots that follow a hardcoded policy.
    The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
    it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
    is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

    When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
    encoding its unique ID, team ID, location, health points and cooldown. A model controlling an agent
    also sees other agents in its visual range (3 × 3 surrounding area). The model gets reward of -1 if the
    team loses or draws at the end of the game. In addition, it also get reward of −0.1 times the total
    health points of the enemy team, which encourages it to attack enemy bots.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(15, 15), n_agents=5, n_opponents=5, init_health=3, full_observable=False,
                 step_cost=0, max_steps=100, step_cool=1):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_opponents = n_opponents
        self._max_steps = max_steps
        self._step_cool = step_cool + 1
        self._step_cost = step_cost
        self._step_count = None

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5 + self._n_opponents) for _ in range(self.n_agents)])

        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}
        self.opp_pos = {_: None for _ in range(self._n_opponents)}
        self.opp_prev_pos = {_: None for _ in range(self._n_opponents)}

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        self.opp_health = {_: None for _ in range(self._n_opponents)}
        self._agent_dones = [None for _ in range(self.n_agents)]
        self._agent_cool = {_: None for _ in range(self.n_agents)}
        self._agent_cool_step = {_: None for _ in range(self.n_agents)}
        self._opp_cool = {_: None for _ in range(self._n_opponents)}
        self._opp_cool_step = {_: None for _ in range(self._n_opponents)}
        self._total_episode_reward = None
        self.viewer = None
        self.full_observable = full_observable

        # 5 * 5 * (type, id, health, cool, x, y)
        self._obs_low = np.repeat(np.array([-1., 0., 0., -1., 0., 0.], dtype=np.float32), 5 * 5)
        self._obs_high = np.repeat(np.array([1., n_opponents, init_health, 1., 1., 1.], dtype=np.float32), 5 * 5)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])
        self.seed()

        # For debug only
        self._agents_trace = {_: None for _ in range(self.n_agents)}
        self._opponents_trace = {_: None for _ in range(self._n_opponents)}

    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(5)]
            meaning += ['Attack Opponent {}'.format(o) for o in range(self._n_opponents)]
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)
            assert agent_i <= self.n_agents

            return action_meaning[agent_i]
        else:
            return action_meaning

    @staticmethod
    def _one_hot_encoding(i, n):
        x = np.zeros(n)
        x[i] = 1
        return x.tolist()

    def get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its team ID, unique ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []
        for agent_i in range(self.n_agents):
            # team id , unique id, location, health, cooldown
            _agent_i_obs = np.zeros((6, 5, 5))
            hp = self.agent_health[agent_i]

            # If agent is alive
            if hp > 0:
                # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
                # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
                # _agent_i_obs += [self.agent_health[agent_i]]
                # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down

                pos = self.agent_pos[agent_i]
                for row in range(0, 5):
                    for col in range(0, 5):
                        if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                                PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                            x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                            _type = 1 if PRE_IDS['agent'] in x else -1
                            _id = int(x[1:]) - 1  # id
                            _agent_i_obs[0][row][col] = _type
                            _agent_i_obs[1][row][col] = _id
                            _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]
                            _agent_i_obs[3][row][col] = self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]
                            _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1  # cool/uncool
                            entity_position = self.agent_pos[_id] if _type == 1 else self.opp_pos[_id]
                            _agent_i_obs[4][row][col] = entity_position[0] / self._grid_shape[0]  # x-coordinate
                            _agent_i_obs[5][row][col] = entity_position[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)
        return _obs

    def get_state(self):
        state = np.zeros((self.n_agents + self._n_opponents, 6))
        # agent info
        for agent_i in range(self.n_agents):
            hp = self.agent_health[agent_i]
            if hp > 0:
                pos = self.agent_pos[agent_i]
                feature = np.array([1, agent_i, hp, 1 if self._agent_cool[agent_i] else -1,
                                    pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]], dtype=np.float)
                state[agent_i] = feature

        # opponent info
        for opp_i in range(self._n_opponents):
            opp_hp = self.opp_health[opp_i]
            if opp_hp > 0:
                pos = self.opp_pos[opp_i]
                feature = np.array([-1, opp_i, opp_hp, 1 if self._opp_cool[opp_i] else -1,
                                    pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]], dtype=np.float)
                state[opp_i + self.n_agents] = feature
        return state.flatten()

    def get_state_size(self):
        return (self.n_agents + self._n_opponents) * 6

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        """ Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
        square around the team center, which is picked uniformly in the grid.
        """
        self._full_obs = self.__create_grid()

        # select agent team center
        # Note : Leaving space from edges so as to have a 5x5 grid around it
        agent_team_center = self.np_random.randint(2, self._grid_shape[0] - 3), self.np_random.randint(2,
                                                                                                       self._grid_shape[
                                                                                                           1] - 3)
        # randomly select agent pos
        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(agent_team_center[0] - 2, agent_team_center[0] + 2),
                       self.np_random.randint(agent_team_center[1] - 2, agent_team_center[1] + 2)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break

        # select opponent team center
        while True:
            pos = self.np_random.randint(2, self._grid_shape[0] - 3), self.np_random.randint(2, self._grid_shape[1] - 3)
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                opp_team_center = pos
                break

        # randomly select opponent pos
        for opp_i in range(self._n_opponents):
            while True:
                pos = [self.np_random.randint(opp_team_center[0] - 2, opp_team_center[0] + 2),
                       self.np_random.randint(opp_team_center[1] - 2, opp_team_center[1] + 2)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.opp_prev_pos[opp_i] = pos
                    self.opp_pos[opp_i] = pos
                    self.__update_opp_view(opp_i)
                    break

        self.__draw_base_img()

    def reset(self):
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        self._agent_cool_step = {_: 0 for _ in range(self.n_agents)}
        self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._opp_cool_step = {_: 0 for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.__init_full_obs()

        # For debug only
        self._agents_trace = {_: [self.agent_pos[_]] for _ in range(self.n_agents)}
        self._opponents_trace = {_: [self.opp_pos[_]] for _ in range(self._n_opponents)}

        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        # draw agents
        for agent_i in range(self.n_agents):
            if self.agent_health[agent_i] > 0:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        # draw opponents
        for opp_i in range(self._n_opponents):
            if self.opp_health[opp_i] > 0:
                fill_cell(img, self.opp_pos[opp_i], cell_size=CELL_SIZE, fill=OPPONENT_COLOR)
                write_cell_text(img, text=str(opp_i + 1), pos=self.opp_pos[opp_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

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
            self.agent_prev_pos[agent_i] = curr_pos
            self.__update_agent_view(agent_i)
            self._agents_trace[agent_i].append(next_pos)

    def __update_opp_pos(self, opp_i, move):

        curr_pos = copy.copy(self.opp_pos[opp_i])
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
            self.opp_pos[opp_i] = next_pos
            self.opp_prev_pos[opp_i] = curr_pos
            self.__update_opp_view(opp_i)
            self._opponents_trace[opp_i].append(next_pos)

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    @staticmethod
    def is_visible(source_pos, target_pos):
        """
        Checks if the target_pos is in the visible range(5x5)  of the source pos

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    @staticmethod
    def is_fireable(source_cooling_down, source_pos, target_pos):
        """
        Checks if the target_pos is in the firing range(5x5)

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return source_cooling_down and (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) \
               and (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)

    def reduce_distance_move(self, opp_i, source_pos, agent_i, target_pos):
        # Todo: makes moves Enum
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append('UP')
        elif source_pos[0] < target_pos[0]:
            _moves.append('DOWN')

        if source_pos[1] > target_pos[1]:
            _moves.append('LEFT')
        elif source_pos[1] < target_pos[1]:
            _moves.append('RIGHT')

        if len(_moves) == 0:
            print(self._step_count, source_pos, target_pos)
            print("agent-{}, hp={}, move_trace={}".format(agent_i, self.agent_health[agent_i],
                                                          self._agents_trace[agent_i]))
            print(
                "opponent-{}, hp={}, move_trace={}".format(opp_i, self.opp_health[opp_i], self._opponents_trace[opp_i]))
            raise AssertionError("One place exists 2 entities!")
        move = self.np_random.choice(_moves)
        for k, v in ACTION_MEANING.items():
            if move.lower() == v.lower():
                move = k
                break
        return move

    @property
    def opps_action(self):
        """
        Opponent bots follow a hardcoded policy.

        The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
        it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
        is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

        :return:
        """

        visible_agents = set([])
        opp_agent_distance = {_: [] for _ in range(self._n_opponents)}

        for opp_i, opp_pos in self.opp_pos.items():
            for agent_i, agent_pos in self.agent_pos.items():
                if agent_i not in visible_agents and self.agent_health[agent_i] > 0 \
                        and self.is_visible(opp_pos, agent_pos):
                    visible_agents.add(agent_i)
                distance = abs(agent_pos[0] - opp_pos[0]) + abs(agent_pos[1] - opp_pos[1])  # manhattan distance
                opp_agent_distance[opp_i].append([distance, agent_i])

        opp_action_n = []
        for opp_i in range(self._n_opponents):
            action = None
            for _, agent_i in sorted(opp_agent_distance[opp_i]):
                if agent_i in visible_agents:
                    if self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i], self.agent_pos[agent_i]):
                        action = agent_i + 5
                    elif self.opp_health[opp_i] > 0:
                        action = self.reduce_distance_move(opp_i, self.opp_pos[opp_i], agent_i, self.agent_pos[agent_i])
                    break
            if action is None:
                if self.opp_health[opp_i] > 0:
                    # logger.debug('No visible agent for enemy:{}'.format(opp_i))
                    action = self.np_random.choice(range(5))
                else:
                    action = 4  # dead opponent could only execute 'no-op' action.
            opp_action_n.append(action)
        return opp_action_n

    def step(self, agents_action):
        assert len(agents_action) == self.n_agents

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # What's the confusion?
        # What if agents attack each other at the same time? Should both of them be effected?
        # Ans: I guess, yes
        # What if other agent moves before the attack is performed in the same time-step.
        # Ans: May be, I can process all the attack actions before move directions to ensure attacks have their effect.

        # processing attacks
        agent_health, opp_health = copy.copy(self.agent_health), copy.copy(self.opp_health)
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action > 4:  # attack actions
                    target_opp = action - 5
                    if self.is_fireable(self._agent_cool[agent_i], self.agent_pos[agent_i], self.opp_pos[target_opp]) \
                            and opp_health[target_opp] > 0:
                        # Fire
                        opp_health[target_opp] -= 1
                        rewards[agent_i] += 1

                        # Update agent cooling down
                        self._agent_cool[agent_i] = False
                        self._agent_cool_step[agent_i] = self._step_cool

                        # Remove opp from the map
                        if opp_health[target_opp] == 0:
                            pos = self.opp_pos[target_opp]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']

                # Update agent cooling down
                self._agent_cool_step[agent_i] = max(self._agent_cool_step[agent_i] - 1, 0)
                if self._agent_cool_step[agent_i] == 0 and not self._agent_cool[agent_i]:
                    self._agent_cool[agent_i] = True

        opp_action = self.opps_action
        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                target_agent = action - 5
                if action > 4:  # attack actions
                    if self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i], self.agent_pos[target_agent]) \
                            and agent_health[target_agent] > 0:
                        # Fire
                        agent_health[target_agent] -= 1
                        rewards[target_agent] -= 1

                        # Update opp cooling down
                        self._opp_cool[opp_i] = False
                        self._opp_cool_step[opp_i] = self._step_cool

                        # Remove agent from the map
                        if agent_health[target_agent] == 0:
                            pos = self.agent_pos[target_agent]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']
                # Update opp cooling down
                self._opp_cool_step[opp_i] = max(self._opp_cool_step[opp_i] - 1, 0)
                if self._opp_cool_step[opp_i] == 0 and not self._opp_cool[opp_i]:
                    self._opp_cool[opp_i] = True

        self.agent_health, self.opp_health = agent_health, opp_health

        # process move actions
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action <= 4:
                    self.__update_agent_pos(agent_i, action)

        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                if action <= 4:
                    self.__update_opp_pos(opp_i, action)

        # step overflow or all opponents dead
        if (self._step_count >= self._max_steps) \
                or (sum([v for k, v in self.opp_health.items()]) == 0) \
                or (sum([v for k, v in self.agent_health.items()]) == 0):
            self._agent_dones = [True for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'health': self.agent_health}

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
OPPONENT_COLOR = 'blue'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A',
    'opponent': 'X',
}
