import copy
import itertools
import logging
from typing import List, Tuple, Union

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

Coordinates = Tuple[int, int]


class Agent:
    """Dataclass keeping all data for one agent/lumberjack in environment.
    In order to keep the support for Python3.6 we are not using `dataclasses` module.

    Attributes:
        id: unique id in one environment run
        pos: position of the agent in grid
    """

    def __init__(self, id: int, pos: Coordinates):
        self.id = id
        self.pos = pos


class Lumberjacks(gym.Env):
    """
    Lumberjacks environment involve a grid world, in which multiple lumberjacks attempt to cut down all trees. In order to cut down a tree in given cell, there must be present greater or equal number of agents/lumberjacks then the tree strength in the same location as tree. Tree is then cut down automatically.

    Agents select one of fire actions ∈ {No-Op, Down, Left, Up, Right}.
    Each agent's observation includes its:
        - agent ID (1)
        - position with in grid (2)
        - number of steps since beginning (1)
        - number of agents and tree strength for each cell in agent view (2 * `np.prod(tuple(2 * v + 1 for v in agent_view))`).
    All values are scaled down into range ∈ [0, 1].

    Only the agents who are involved in cutting down the tree are rewarded with `tree_cutdown_reward`.
    The environment is terminated as soon as all trees are cut down or when the number of steps reach the `max_steps`.

    Upon rendering, we show the grid, where each cell shows the agents (blue) and tree (green) with their current strength.

    Args:
        grid_shape: size of the grid
        n_agents: number of agents/lumberjacks
        n_trees: number of trees
        agent_view: size of the agent view range in each direction
        full_observable: flag whether agents should receive observation for all other agents
        step_cost: reward receive in each time step
        tree_cutdown_reward: reward received by agents who cut down the tree
        max_steps: maximum steps in one environment episode

    Attributes:
        _agents: list of all agents. The index in this list is also the ID of the agent
        _agent_map: tree dimensional numpy array of indicators where the agents are located
        _tree_map: two dimensional numpy array of strength of the trees
        _total_episode_reward: array with accumulated rewards for each agent.
        _agent_dones: list with indicater whether the agent is done or not.
        _base_img: base image with grid
        _viewer: viewer for the rendered image
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape: Coordinates = (5, 5), n_agents: int = 2, n_trees: int = 12,
                 agent_view: Tuple[int, int] = (1, 1), full_observable: bool = False,
                 step_cost: float = -1, tree_cutdown_reward: float = 10, max_steps: int = 100):
        assert 0 < n_agents
        assert n_agents + n_trees <= np.prod(grid_shape)
        assert 1 <= agent_view[0] <= grid_shape[0] and 1 <= agent_view[1] <= grid_shape[1]

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_trees = n_trees
        self._agent_view = agent_view
        self.full_observable = full_observable
        self._step_cost = step_cost
        self._tree_cutdown_reward = tree_cutdown_reward
        self._max_steps = max_steps
        self.steps_beyond_done = 0
        self.seed()

        self._agents = []  # List[Agent]
        # In order to speed up the environment we used the advantage of vector operations.
        # Therefor we need to pad the grid size by the maximum agent_view size.
        # Relative coordinates refer to the coordinates in non pad grid. These are the only
        # coordinates visible to user. Extended coordinates refer to the coordinates in pad grid.
        self._agent_map = None
        self._tree_map = None
        self._total_episode_reward = None
        self._agent_dones = None

        mask_size = np.prod(tuple(2 * v + 1 for v in self._agent_view))
        # Agent ID (1) + Pos (2) + Step (1) + Neighborhood (2 * mask_size)
        self._obs_len = (1 + 2 + 1 + 2 * mask_size)
        obs_high = np.array([1.] * self._obs_len, dtype=np.float32)
        obs_low = np.array([0.] * self._obs_len, dtype=np.float32)
        if self.full_observable:
            obs_high = np.tile(obs_high, self.n_agents)
            obs_low = np.tile(obs_low, self.n_agents)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(5)] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(obs_low, obs_high)] * self.n_agents)

        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        self._viewer = None

    def get_action_meanings(self, agent_id: int = None) -> Union[List[str], List[List[str]]]:
        """Returns list of actions meaning for `agent_id`.

        If `agent_id` is not specified returns meaning for all agents.
        """
        if agent_id is not None:
            assert agent_id <= self.n_agents
            return [k.upper() for k, v in sorted(ACTIONS_IDS.items(), key=lambda item: item[1])]
        else:
            return [[k.upper() for k, v in sorted(ACTIONS_IDS.items(), key=lambda item: item[1])]]

    def reset(self) -> List[List[float]]:
        self._init_episode()
        self._step_count = 0
        self._total_episode_reward = np.zeros(self.n_agents)
        self._agent_dones = [False] * self.n_agents
        self.steps_beyond_done = 0

        return self.get_agent_obs()

    def _init_episode(self):
        """Initialize environment for new episode.

        Fills `self._agents`, self._agent_map` and `self._tree_map` with new values.
        """
        init_positions = self._generate_init_pos()
        agent_id, tree_id = 0, self.n_agents
        self._agents = []
        self._agent_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
            self.n_agents
        ), dtype=np.int32)
        self._tree_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)

        for pos, cell in np.ndenumerate(init_positions):
            pos = self._to_extended_coordinates(pos)
            if cell == PRE_IDS['agent']:
                self._agent_map[pos[0], pos[1], agent_id] = 1
                self._agents.append(Agent(agent_id, pos=pos))
                agent_id += 1
            elif cell == PRE_IDS['tree']:
                self._tree_map[pos] = self.np_random.randint(1, self.n_agents + 1)
                tree_id += 1

    def _to_extended_coordinates(self, relative_coordinates):
        """Translate relative coordinates in to the extended coordinates."""
        return relative_coordinates[0] + self._agent_view[0], relative_coordinates[1] + self._agent_view[1]

    def _to_relative_coordinates(self, extended_coordinates):
        """Translate extended coordinates in to the relative coordinates."""
        return extended_coordinates[0] - self._agent_view[0], extended_coordinates[1] - self._agent_view[1]

    def _generate_init_pos(self) -> np.ndarray:
        """Returns randomly selected initial positions for agents and trees in relative coordinates.

        No agent or trees share the same cell in initial positions.
        """
        init_pos = np.array(
            [PRE_IDS['agent']] * self.n_agents +
            [PRE_IDS['tree']] * self._n_trees +
            [PRE_IDS['empty']] * (np.prod(self._grid_shape) - self.n_agents - self._n_trees)
        )
        self.np_random.shuffle(init_pos)
        return np.reshape(init_pos, self._grid_shape)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        mask = (
            slice(self._agent_view[0], self._agent_view[0] + self._grid_shape[0]),
            slice(self._agent_view[1], self._agent_view[1] + self._grid_shape[1]),
        )

        # Iterate over all grid positions
        for pos, agent_strength, tree_strength in self._view_generator(mask):
            if tree_strength and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                tree_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            else:
                cell_size = (CELL_SIZE, CELL_SIZE)
                tree_pos = agent_pos = (pos[0], pos[1])

            if tree_strength != 0:
                fill_cell(img, pos=tree_pos, cell_size=cell_size, fill=TREE_COLOR, margin=0.1)
                write_cell_text(img, text=str(tree_strength), pos=tree_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if agent_strength != 0:
                draw_circle(img, pos=agent_pos, cell_size=cell_size, fill=AGENT_COLOR, radius=0.30)
                write_cell_text(img, text=str(agent_strength), pos=agent_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen

    def _view_generator(self, mask: Tuple[slice, slice]) -> Tuple[Coordinates, int, int]:
        """Yields position, number of agent and tree strength for all cells defined by `mask`.

        Args:
            mask: tuple of slices in extended coordinates.
        """
        agent_iter = np.ndenumerate(np.sum(self._agent_map[mask], axis=2))
        tree_iter = np.nditer(self._tree_map[mask])
        for (pos, n_a), n_t in zip(agent_iter, tree_iter):
            yield pos, n_a, n_t

    def get_agent_obs(self) -> List[List[float]]:
        """Returns list of observations for each agent."""
        obs = np.zeros((self.n_agents, self._obs_len))
        for i, (agent_id, agent) in enumerate(self._agent_generator()):
            rel_pos = self._to_relative_coordinates(agent.pos)
            obs[i, 0] = agent_id / self.n_agents  # Agent ID
            obs[i, 1] = rel_pos[0] / (self._grid_shape[0] - 1)  # Coordinate
            obs[i, 2] = rel_pos[1] / (self._grid_shape[1] - 1)  # Coordinate
            obs[i, 3] = self._step_count / self._max_steps  # Steps

            for j, (_, agent_strength, tree_strength) in zip(
                    itertools.count(start=4, step=2),
                    self._agent_view_generator(agent.pos, self._agent_view)):
                obs[i, j] = agent_strength / self.n_agents
                obs[i, j + 1] = tree_strength / self.n_agents

        # Convert it from numpy array
        obs = obs.tolist()

        if self.full_observable:
            obs = [feature for agent_obs in obs for feature in agent_obs]
            obs = [obs] * self.n_agents

        return obs

    def _agent_generator(self) -> Tuple[int, Agent]:
        """Yields agent_id and agent for all agents in environment."""
        for agent_id, agent in enumerate(self._agents):
            yield agent_id, agent

    def _agent_view_generator(self, pos: Coordinates, view_range: Tuple[int, int]):
        """Yields position, number of agent and tree strength for cells in distance of `view_range` from `pos`.  """
        mask = (
            slice(pos[0] - view_range[0], pos[0] + view_range[0] + 1),
            slice(pos[1] - view_range[1], pos[1] + view_range[1] + 1),
        )
        yield from self._view_generator(mask)

    def step(self, agents_action: List[int]):
        # Assert would slow down the environment which is undesirable. We rather expect the check on the user side.
        # assert len(agents_action) == self.n_agents

        # Following snippet of code was refereed from:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L124
        if all(self._agent_dones):
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            return self.get_agent_obs(), [0] * self.n_agents, self._agent_dones, {}

        self._step_count += 1
        rewards = np.full(self.n_agents, self._step_cost)

        # Move agents
        for (agent_id, agent), action in zip(self._agent_generator(), agents_action):
            if not self._agent_dones[agent_id]:
                self._update_agent_pos(agent, action)

        # Cut down trees
        mask = (np.sum(self._agent_map, axis=2) >= self._tree_map) & (self._tree_map > 0)
        self._tree_map[mask] = 0

        # Calculate rewards
        rewards += np.sum(mask * self._tree_cutdown_reward, axis=(0, 1))
        self._total_episode_reward += rewards

        if (self._step_count >= self._max_steps) or (np.count_nonzero(self._tree_map) == 0):
            self._agent_dones = [True] * self.n_agents

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def _update_agent_pos(self, agent: Agent, move: int):
        """Moves `agent` according the `move` command."""
        next_pos = self._next_pos(agent.pos, move)

        # Remove agent from old position
        self._agent_map[agent.pos[0], agent.pos[1], agent.id] = 0

        # Add agent to the new position
        agent.pos = next_pos
        self._agent_map[next_pos[0], next_pos[1], agent.id] = 1

    def _next_pos(self, curr_pos: Coordinates, move: int) -> Coordinates:
        """Returns next valid position in extended coordinates given by `move` command relative to `curr_pos`."""
        if move == ACTIONS_IDS['noop']:
            next_pos = curr_pos
        elif move == ACTIONS_IDS['down']:
            next_pos = (curr_pos[0] + 1, curr_pos[1])
        elif move == ACTIONS_IDS['left']:
            next_pos = (curr_pos[0], curr_pos[1] - 1)
        elif move == ACTIONS_IDS['up']:
            next_pos = (curr_pos[0] - 1, curr_pos[1])
        elif move == ACTIONS_IDS['right']:
            next_pos = (curr_pos[0], curr_pos[1] + 1)
        else:
            raise ValueError('Unknown action {}. Valid action are {}'.format(move, list(ACTIONS_IDS.values())))
        # np.clip is significantly slower, see: https://github.com/numpy/numpy/issues/14281
        # return tuple(np.clip(next_pos,
        #                      (self._agent_view[0], self._agent_view[1]),
        #                      (self._agent_view[0] + self._grid_shape[0] - 1,
        #                       self._agent_view[1] + self._grid_shape[1] - 1),
        #                      ))
        return (
            min(max(next_pos[0], self._agent_view[0]), self._grid_shape[0] - 1),
            min(max(next_pos[1], self._agent_view[1]), self._grid_shape[1] - 1),
        )

    def seed(self, n: Union[None, int] = None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
TREE_COLOR = 'green'
WALL_COLOR = 'black'

CELL_SIZE = 35

ACTIONS_IDS = {
    'noop': 0,
    'down': 1,
    'left': 2,
    'up': 3,
    'right': 4,
}

PRE_IDS = {
    'empty': 0,
    'wall': 1,
    'agent': 2,
    'tree': 3,
}
