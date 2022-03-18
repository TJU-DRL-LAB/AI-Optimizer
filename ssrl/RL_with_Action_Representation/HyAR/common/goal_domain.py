import numpy as np
import sys
import os
import gym
import gym_goal
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH

sys.path.append(os.path.abspath(os.path.pardir))
from agents.basis import ScaledBasis  # TODO: fix relative import


FOURIER_DIM = 7


class CustomFourierBasis(ScaledBasis):
    """
    Custom Fourier basis from Warwick's code
    """

    def __init__(self, nvars, low, high):
        super().__init__(nvars, low, high, False)
        assert nvars == 14
        self.coeffs = []
        self.__generate_coefficients(self.coeffs)
        self.num_terms = len(self.coeffs)
        self._shrink = np.ones(self.num_terms)
        for i in range(1, self.num_terms):
            self._shrink[i] = np.linalg.norm(self.coeffs[i])

    def compute_features(self, state):
        scaled_state = super().compute_features(state)
        return np.cos(np.dot(self.coeffs, scaled_state))

    def __generate_coefficients(self, coeffs, vector=np.zeros((14,)), depth=0, count=0):
        """ Generate all coefficient vectors. """
        if depth == 14 or count == 2:
            coeffs.append(vector)
        else:
            if depth in [3, 4, 7, 8, 12, 13]:
                self.__generate_coefficients(coeffs, vector, depth + 1, count)
            else:
                for j in range(FOURIER_DIM):
                    new_vector = np.copy(vector)
                    new_vector[depth] = np.pi * j
                    self.__generate_coefficients(coeffs, new_vector, depth + 1, count + (j > 0))


class GoalFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(GoalFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action


class GoalObservationWrapper(gym.ObservationWrapper):
    """
    Extends the Goal domain state with keeper and ball difference features.
    """

    def __init__(self, env):
        super(GoalObservationWrapper, self).__init__(env)
        base_state = env.get_state()
        ball_feats = self.ball_features(base_state)
        keeper_feats = self.keeper_features(base_state)
        newshape = (base_state.shape[0] + ball_feats.shape[0] + keeper_feats.shape[0],)
        low = np.zeros(newshape)
        low[:14] = env.observation_space.spaces[0].low
        # since keeper-ball difference vector is normalised
        low[14] = -1.
        low[15] = -1.
        low[16] = -GOAL_WIDTH / 2
        high = np.ones(newshape)
        high[:14] = env.observation_space.spaces[0].high
        # since keeper-ball difference vector is normalised
        high[14] = 1.
        high[15] = 1.
        high[16] = GOAL_WIDTH
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=low, high=high, dtype=np.float32),
            gym.spaces.Discrete(200),  # steps (200 limit is an estimate)
        ))

    @staticmethod
    def keeper_projection(state):
        if state[5] == state[10]:
            if state[6] < state[11]:
                return -GOAL_WIDTH / 2
            else:
                return GOAL_WIDTH / 2
        grad = (state[6] - state[11]) / (state[5] - state[10])
        y_int = state[11] - grad * state[10]
        pos = grad * PITCH_LENGTH / 2 + y_int
        return np.clip(pos, -GOAL_WIDTH / 2, GOAL_WIDTH)

    def keeper_features(self, state):
        """
        Returns [g], where g is the projection
        of the goalie onto the goal line.
        """
        _state = state
        yval = self.keeper_projection(_state)
        return np.array([yval])

    @staticmethod
    def position_features(state):
        """
        Returns [1 p p^2], containing the squared features
        of the player position.
        """
        xval = state[0] / (PITCH_LENGTH / 2)
        yval = state[1] / (PITCH_WIDTH / 2)
        return np.array([1., xval, yval, xval ** 2, yval ** 2])

    def ball_features(self, state):
        """ Returns ball-based position features. """
        ball = np.array((state[10], state[11]))
        keeper = np.array((state[5], state[6]))
        diff = (ball - keeper) / np.linalg.norm(ball - keeper)
        return np.array([diff[0], diff[1]])

    def observation(self, obs):
        state, steps = obs
        state = np.concatenate((state, self.ball_features(state), self.keeper_features(state)))
        return (state, steps)
