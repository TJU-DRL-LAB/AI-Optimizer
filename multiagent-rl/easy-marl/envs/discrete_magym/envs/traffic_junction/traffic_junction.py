# -*- coding: utf-8 -*-

import copy
import logging
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class TrafficJunction(gym.Env):
    """
    This consists of a 4-way junction on a 14 × 14 grid. At each time step, "new" cars enter the grid with
    probability `p_arrive` from each of the four directions. However, the total number of cars at any given
    time is limited to `Nmax`.

    Each car occupies a single cell at any given time and is randomly assigned to one of three possible routes
    (keeping to the right-hand side of the road). At every time step, a car has two possible actions: gas which advances
    it by one cell on its route or brake to stay at its current location. A car will be removed once it reaches its
    destination at the edge of the grid.

    Two cars collide if their locations overlap. A collision incurs a reward `rcoll = −10`, but does not affect
    the simulation in any other way. To discourage a traffic jam, each car gets reward of `τ * r_time = −0.01τ`
    at every time step, where `τ` is the number time steps passed since the car arrived. Therefore, the total
    reward at time t is

    r(t) = C^t * r_coll + \sum_{i=1}_{N^t} {\tau_i * r_time}

    where C^t is the number of collisions occurring at time t and N^t is number of cars present. The simulation is
    terminated after 'max_steps(default:40)' steps and is classified as a failure if one or more collisions have
    occurred.

    Each car is represented by one-hot binary vector set {n, l, r}, that encodes its unique ID, current location
    and assigned route number respectively. Each agent controlling a car can only observe other cars in its vision
    range (a surrounding 3 × 3 neighborhood), though low level communication is allowed in "v1" version of the game.

    The state vector s_j for each agent is thus a concatenation of all these vectors, having dimension
    (3^2) × (|n| + |l| + |r|).

    Reference : Learning Multi-agent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf


    For details on various versions, please refer to "wiki"
    (https://github.com/koulanurag/ma-gym/wiki/Environments#TrafficJunction)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(14, 14), step_cost=-0.01, n_max=4, collision_reward=-10, arrive_prob=0.5,
                 full_observable: bool = False, max_steps: int = 100):
        assert 1 <= n_max <= 10, "n_max should be range in [1,10]"
        assert 0 <= arrive_prob <= 1, "arrive probability should be in range [0,1]"
        assert len(grid_shape) == 2, 'only 2-d grids are acceptable'
        assert 1 <= max_steps, "max_steps should be more than 1"

        self._grid_shape = grid_shape
        self.n_agents = n_max
        self._max_steps = max_steps
        self._step_count = 0  # environment step counter
        self._collision_reward = collision_reward
        self._total_episode_reward = None
        self._arrive_prob = arrive_prob
        self._n_max = n_max
        self._step_cost = step_cost
        self.curr_cars_count = 0
        self._n_routes = 3

        self._agent_view_mask = (3, 3)

        # entry gates where the cars spawn
        # Note: [(7, 0), (13, 7), (6, 13), (0, 6)] for (14 x 14) grid
        self._entry_gates = [(self._grid_shape[0] // 2, 0),
                             (self._grid_shape[0] - 1, self._grid_shape[1] // 2),
                             (self._grid_shape[0] // 2 - 1, self._grid_shape[1] - 1),
                             (0, self._grid_shape[1] // 2 - 1)]

        # destination places for the cars to reach
        # Note: [(7, 13), (0, 7), (6, 0), (13, 6)] for (14 x 14) grid
        self._destination = [(self._grid_shape[0] // 2, self._grid_shape[1] - 1),
                             (0, self._grid_shape[1] // 2),
                             (self._grid_shape[0] // 2 - 1, 0),
                             (self._grid_shape[0] - 1, self._grid_shape[1] // 2 - 1)]

        # dict{direction_vectors: (turn_right, turn_left)}
        # Note: [((7, 6), (7,7))), ((7, 7),(6,7)), ((6,6),(7, 6)), ((6, 7),(6,6))] for (14 x14) grid
        self._turning_places = {(0, 1): ((self._grid_shape[0] // 2, self._grid_shape[0] // 2 - 1),
                                         (self._grid_shape[0] // 2, self._grid_shape[0] // 2)),
                                (-1, 0): ((self._grid_shape[0] // 2, self._grid_shape[0] // 2),
                                          (self._grid_shape[0] // 2 - 1, self._grid_shape[0] // 2)),
                                (1, 0): ((self._grid_shape[0] // 2 - 1, self._grid_shape[0] // 2 - 1),
                                         (self._grid_shape[0] // 2, self._grid_shape[0] // 2 - 1)),
                                (0, -1): ((self._grid_shape[0] // 2 - 1, self._grid_shape[0] // 2),
                                          (self._grid_shape[0] // 2 - 1, self._grid_shape[0] // 2 - 1))}

        # dict{starting_place: direction_vector}
        self._route_vectors = {(self._grid_shape[0] // 2, 0): (0, 1),
                               (self._grid_shape[0] - 1, self._grid_shape[0] // 2): (-1, 0),
                               (0, self._grid_shape[0] // 2 - 1): (1, 0),
                               (self._grid_shape[0] // 2 - 1, self._grid_shape[0] - 1): (0, -1)}

        self._agent_turned = [False for _ in range(self.n_agents)]  # flag if car changed direction
        self._agents_routes = [-1 for _ in range(self.n_agents)]  # route each car is following atm
        self._agents_direction = [(0, 0) for _ in range(self.n_agents)]  # cars are not on the road initially
        self._agent_step_count = [0 for _ in range(self.n_agents)]  # holds a step counter for each car

        self.action_space = MultiAgentActionSpace([spaces.Discrete(2) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self._on_the_road = [False for _ in range(self.n_agents)]  # flag if car is on the road

        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self._agent_dones = [None for _ in range(self.n_agents)]

        self.viewer = None
        self.full_observable = full_observable

        # agent id (n_agents, onehot), obs_mask (9), pos (2), route (3)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.ones((mask_size * (self.n_agents + self._n_routes + 2)))  # 2 is for location
        self._obs_low = np.zeros((mask_size * (self.n_agents + self._n_routes + 2)))  # 2 is for location
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __init_full_obs(self):
        """
        Initiates environment: inserts up to |entry_gates| cars. once the entry gates are filled, the remaining agents
        stay initialized outside the road waiting to enter
        """
        self._full_obs = self.__create_grid()

        shuffled_gates = list(self._route_vectors.keys())
        random.shuffle(shuffled_gates)
        for agent_i in range(self.n_agents):
            if self.curr_cars_count >= len(self._entry_gates):
                self.agent_pos[agent_i] = (0, 0)  # not yet on the road
            else:
                pos = shuffled_gates[agent_i]
                # gets direction vector for agent_i that spawned in position pos
                self._agents_direction[agent_i] = self._route_vectors[pos]
                self.agent_pos[agent_i] = pos
                self.curr_cars_count += 1
                self._on_the_road[agent_i] = True
                self._agents_routes[agent_i] = random.randint(1, self._n_routes)  # [1,3] (inclusive)
            self.__update_agent_view(agent_i)

        self.__draw_base_img()

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __check_collision(self, pos):
        """
        Verifies if a transition to the position pos will result on a collision.
        :param pos: position to verify if there is collision
        :type pos: tuple

        :return: boolean stating true or false
        :rtype: bool
        """
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]].find(PRE_IDS['agent']) > -1)

    def __is_gate_free(self):
        """
        Verifies if any spawning gate is free for a car to be placed

        :return: list of currently free gates
        :rtype: list
        """
        free_gates = []
        for pos in self._entry_gates:
            if pos not in self.agent_pos.values():
                free_gates.append(pos)
        return free_gates

    def __reached_dest(self, agent_i):
        """
        Verifies if the agent_i reached a destination place.
        :param agent_i: id of the agent
        :type agent_i: int

        :return: boolean stating true or false
        :rtype: bool  
        """
        pos = self.agent_pos[agent_i]
        if pos in self._destination:
            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']
            return True
        return False

    def get_agent_obs(self):
        """
        Computes the observations for the agents. Each agent receives information about cars in it's vision
        range (a surrounding 3 × 3 neighborhood),where each car is represented by one-hot binary vector set {n, l, r},
        that encodes its unique ID, current location and assigned route number respectively.

        The state vector s_j for each agent is thus a concatenation of all these vectors, having dimension
        (3^2) × (|n| + |l| + |r|).

        :return: list with observations of all agents. the full list has shape (n_agents, (3^2) × (|n| + |l| + |r|))
        :rtype: list
        """
        agent_no_mask_obs = []

        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # agent id
            _agent_i_obs = [0 for _ in range(self.n_agents)]
            _agent_i_obs[agent_i] = 1

            # location
            _agent_i_obs += [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # route 
            route_agent_i = np.zeros(self._n_routes)
            route_agent_i[self._agents_routes[agent_i] - 1] = 1

            _agent_i_obs += route_agent_i.tolist()

            agent_no_mask_obs.append(_agent_i_obs)

        agent_obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            mask_view = np.zeros((*self._agent_view_mask, len(agent_no_mask_obs[0])))
            for row in range(max(0, pos[0] - 1), min(pos[0] + 1 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 1), min(pos[1] + 1 + 1, self._grid_shape[1])):
                    if PRE_IDS['agent'] in self._full_obs[row][col]:
                        _id = int(self._full_obs[row][col].split(PRE_IDS['agent'])[1]) - 1
                        mask_view[row - (pos[0] - 1), col - (pos[1] - 1), :] = agent_no_mask_obs[_id]
            agent_obs.append(mask_view.flatten())

        if self.full_observable:
            _obs = np.array(agent_obs).flatten().tolist()
            agent_obs = [_obs for _ in range(self.n_agents)]
        return agent_obs

    def __draw_base_img(self):
        # create grid and make everything black
        img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=WALL_COLOR)

        # draw tracks
        for i, row in enumerate(self._full_obs):
            for j, col in enumerate(row):
                if col == PRE_IDS['empty']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill=(143, 141, 136), margin=0.05)
                elif col == PRE_IDS['wall']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill=(242, 227, 167), margin=0.02)

        return img

    def __create_grid(self):
        # create a grid with every cell as wall
        _grid = [[PRE_IDS['wall'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]

        # draw track by making cells empty :
        # horizontal tracks
        _grid[self._grid_shape[0] // 2 - 1] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]
        _grid[self._grid_shape[0] // 2] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]

        # vertical tracks
        for row in range(self._grid_shape[0]):
            _grid[row][self._grid_shape[1] // 2 - 1] = PRE_IDS['empty']
            _grid[row][self._grid_shape[1] // 2] = PRE_IDS['empty']

        return _grid

    def step(self, agents_action):
        """
        Performs an action in the environment and steps forward. At each step a new agent enters the road by
        one of the 4 gates according to a probability "_arrive_prob". A "ncoll" reward is given to an agent if it
        collides and all of them receive "-0.01*step_n" to avoid traffic jams.

        :param agents_action: list of actions of all the agents to perform in the environment
        :type agents_action: list

        :return: agents observations, rewards, if agents are done and additional info
        :rtype: tuple
        """
        assert len(agents_action) == self.n_agents, \
            "Invalid action! It was expected to be list of {}" \
            " dimension but was found to be of {}".format(self.n_agents, len(agents_action))

        assert all([action_i in ACTION_MEANING.keys() for action_i in agents_action]), \
            "Invalid action found in the list of sampled actions {}" \
            ". Valid actions are {}".format(agents_action, ACTION_MEANING.keys())

        self._step_count += 1  # global environment step
        rewards = [0 for _ in range(self.n_agents)]  # initialize rewards array
        step_collisions = 0  # counts collisions in this step

        # checks if there is a collision; this is done in the __update_agent_pos method
        # we still need to check both agent_dones and on_the_road because an agent may not be done
        # and have not entered the road yet 
        for agent_i, action in enumerate(agents_action):
            if not self._agent_dones[agent_i] and self._on_the_road[agent_i]:
                self._agent_step_count[agent_i] += 1  # agent step count
                collision_flag = self.__update_agent_pos(agent_i, action)
                if collision_flag:
                    rewards[agent_i] += self._collision_reward
                    step_collisions += 1

                # gives additional step punishment to avoid jams
                # at every time step, where `τ` is the number time steps passed since the car arrived.
                # We need to keep track of step_count of each car and that has to be multiplied.
                rewards[agent_i] += self._step_cost * self._agent_step_count[agent_i]
            self._total_episode_reward[agent_i] += rewards[agent_i]

            # checks if destination was reached
            # once a car reaches it's destination , it will never enter again in any of the tracks
            # Also, if all cars have reached their destination, then we terminate the episode.
            if self.__reached_dest(agent_i):
                self._agent_dones[agent_i] = True
                self.curr_cars_count -= 1

            # if max_steps was reached, terminate the episode
            if self._step_count >= self._max_steps:
                self._agent_dones[agent_i] = True

        # adds new car according to the probability _arrive_prob
        if random.uniform(0, 1) < self._arrive_prob:
            free_gates = self.__is_gate_free()
            # if there are agents outside the road and if any gate is free
            if not all(self._on_the_road) and free_gates:
                # then gets first agent on the list which is not on the road
                agent_to_enter = self._on_the_road.index(False)
                pos = random.choice(free_gates)
                self._agents_direction[agent_to_enter] = self._route_vectors[pos]
                self.agent_pos[agent_to_enter] = pos
                self.curr_cars_count += 1
                self._on_the_road[agent_to_enter] = True
                self._agent_turned[agent_to_enter] = False
                self._agents_routes[agent_to_enter] = random.randint(1, self._n_routes)  # (1, 3)
                self.__update_agent_view(agent_to_enter)

        return self.get_agent_obs(), rewards, self._agent_dones, {'step_collisions': step_collisions}

    def __get_next_direction(self, route, agent_i):
        """
        Computes the new direction vector after the cars turn on the junction for route 2 (turn right) and 3 (turn left)
        :param route: route that was assigned to the car (1 - fwd, 2 - turn right, 3 - turn left)
        :type route: int

        :param agent_i: id of the agent
        :type agent_i: int

        :return: new direction vector following the assigned route
        :rtype: tuple
        """
        # gets current direction vector
        dir_vector = self._agents_direction[agent_i]

        sig = (1 if dir_vector[1] != 0 else -1) if route == 2 else (-1 if dir_vector[1] != 0 else 1)
        new_dir_vector = (dir_vector[1] * sig, 0) if dir_vector[0] == 0 else (0, dir_vector[0] * sig)

        return new_dir_vector

    def __update_agent_pos(self, agent_i, move):
        """
        Updates the agent position in the environment. Moves can be 0 (GAS) or 1 (BRAKE). If the move is 1 does nothing,
        car remains stopped. If the move is 0 then evaluate the route assigned. If the route is 1 (forward) then
        maintain the same direction vector. Otherwise, compute new direction vector and apply the change of direction
        when the junction turning place was reached. After the move is made, verifies if it resulted into a collision
        and returns the reward collision if that happens. The position is only updated if no collision occurred.

        :param agent_i: id of the agent
        :type agent_i: int

        :param move: move picked by the agent_i
        :type move: int

        :return: bool flag associated to the existence or absence of a collision
        :rtype: bool
        """

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        route = self._agents_routes[agent_i]

        if move == 0:  # GAS
            if route == 1:
                next_pos = tuple([curr_pos[i] + self._agents_direction[agent_i][i] for i in range(len(curr_pos))])
            else:
                turn_pos = self._turning_places[self._agents_direction[agent_i]]
                # if the car reached the turning position in the junction for his route and starting gate
                if curr_pos == turn_pos[route - 2] and not self._agent_turned[agent_i]:
                    new_dir_vector = self.__get_next_direction(route, agent_i)
                    self._agents_direction[agent_i] = new_dir_vector
                    self._agent_turned[agent_i] = True
                    next_pos = tuple([curr_pos[i] + new_dir_vector[i] for i in range(len(curr_pos))])
                else:
                    next_pos = tuple([curr_pos[i] + self._agents_direction[agent_i][i] for i in range(len(curr_pos))])
        elif move == 1:  # BRAKE
            pass
        else:
            raise Exception('Action Not found!')

        # if there is a collision
        if next_pos is not None and self.__check_collision(next_pos):
            return True

        # if there is no collision and the next position is free updates agent position
        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

        return False

    def reset(self):
        """
        Resets the environment when a terminal state is reached. 

        :return: list with the observations of the agents
        :rtype: list
        """
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self._step_count = 0
        self._agent_step_count = [0 for _ in range(self.n_agents)]
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._on_the_road = [False for _ in range(self.n_agents)]
        self._agent_turned = [False for _ in range(self.n_agents)]
        self.curr_cars_count = 0

        self.agent_pos = {}
        self.__init_full_obs()

        return self.get_agent_obs()

    def render(self, mode: str = 'human'):
        img = copy.copy(self._base_img)

        for agent_i in range(self.n_agents):
            if not self._agent_dones[agent_i] and self._on_the_road[agent_i]:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENTS_COLORS[agent_i])
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
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

    def seed(self, n: int):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 30

WALL_COLOR = 'black'

# fixed colors for #agents = n_max <= 10
AGENTS_COLORS = [
    "red",
    "blue",
    "yellow",
    "orange",
    "black",
    "green",
    "purple",
    "pink",
    "brown",
    "grey"
]

ACTION_MEANING = {
    0: "GAS",
    1: "BRAKE",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A'
}
