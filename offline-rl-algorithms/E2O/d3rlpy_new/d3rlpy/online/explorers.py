from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
from typing_extensions import Protocol

from ..preprocessing.action_scalers import ActionScaler, MinMaxActionScaler


class _ActionProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    @property
    def action_size(self) -> Optional[int]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...


class Explorer(metaclass=ABCMeta):
    @abstractmethod
    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        pass


class ConstantEpsilonGreedy(Explorer):
    """:math:`\\epsilon`-greedy explorer with constant :math:`\\epsilon`.

    Args:
        epsilon (float): the constant :math:`\\epsilon`.

    """

    _epsilon: float

    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        greedy_actions = algo.predict(x)
        random_actions = np.random.randint(algo.action_size, size=x.shape[0])
        is_random = np.random.random(x.shape[0]) < self._epsilon
        return np.where(is_random, random_actions, greedy_actions)


class LinearDecayEpsilonGreedy(Explorer):
    """:math:`\\epsilon`-greedy explorer with linear decay schedule.

    Args:
        start_epsilon (float): the beginning :math:`\\epsilon`.
        end_epsilon (float): the end :math:`\\epsilon`.
        duration (int): the scheduling duration.

    """

    _start_epsilon: float
    _end_epsilon: float
    _duration: int

    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.1,
        duration: int = 1000000,
    ):
        self._start_epsilon = start_epsilon
        self._end_epsilon = end_epsilon
        self._duration = duration

    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        """Returns :math:`\\epsilon`-greedy action.

        Args:
            algo: algorithm.
            x: observation.
            step: current environment step.

        Returns:
            :math:`\\epsilon`-greedy action.

        """
        greedy_actions = algo.predict(x)
        random_actions = np.random.randint(algo.action_size, size=x.shape[0])
        is_random = np.random.random(x.shape[0]) < self.compute_epsilon(step)
        return np.where(is_random, random_actions, greedy_actions)

    def compute_epsilon(self, step: int) -> float:
        """Returns decayed :math:`\\epsilon`.

        Returns:
            :math:`\\epsilon`.

        """
        if step >= self._duration:
            return self._end_epsilon
        base = self._start_epsilon - self._end_epsilon
        return base * (1.0 - step / self._duration) + self._end_epsilon


class NormalNoise(Explorer):
    """Normal noise explorer.

    Args:
        mean (float): mean.
        std (float): standard deviation.

    """

    _mean: float
    _std: float

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self._mean = mean
        self._std = std

    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        """Returns action with noise injection.

        Args:
            algo: algorithm.
            x: observation.

        Returns:
            action with noise injection.

        """
        action = algo.predict(x)
        # FIXME: Scalar noise works better for some reason.
        #        But this is different from paper.
        noise = np.random.normal(self._mean, self._std)

        if isinstance(algo.action_scaler, MinMaxActionScaler):
            # scale noise
            params = algo.action_scaler.get_params()
            minimum = params["minimum"]
            maximum = params["maximum"]
        else:
            minimum = -1.0
            maximum = 1.0

        return np.clip(action + noise, minimum, maximum)
