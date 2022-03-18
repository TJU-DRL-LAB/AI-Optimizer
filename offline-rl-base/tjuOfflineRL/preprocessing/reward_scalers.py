from typing import Any, ClassVar, Dict, List, Optional, Type

import gym
import numpy as np
import torch

from ..dataset import MDPDataset, Transition
from ..decorators import pretty_repr
from ..logger import LOG


@pretty_repr
class RewardScaler:

    TYPE: ClassVar[str] = "none"

    def fit(self, transitions: List[Transition]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            transitions: list of transitions.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
        """Gets scaling parameters from environment.

        Note:
            ``RewardScaler`` does not support fitting with environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError("Please initialize with dataset.")

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        """Returns processed rewards.

        Args:
            reward: reward.

        Returns:
            processed reward.

        """
        raise NotImplementedError

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        """Returns reversely processed rewards.

        Args:
            reward: reward.

        Returns:
            reversely processed reward.

        """
        raise NotImplementedError

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        """Returns transformed rewards in numpy array.

        Args:
            reward: reward.

        Returns:
            transformed reward.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns a scaler type.

        Returns:
            scaler type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns scaling parameters.

        Args:
            deep: flag to deeply copy objects.

        Returns:
            scaler parameters.

        """
        raise NotImplementedError


class MultiplyRewardScaler(RewardScaler):
    r"""Multiplication reward preprocessing.

    This preprocessor multiplies rewards by a constant number.

    .. code-block:: python

        from tjuOfflineRL.preprocessing import MultiplyRewardScaler

        # multiply rewards by 10
        reward_scaler = MultiplyRewardScaler(10.0)

        cql = CQL(reward_scaler=reward_scaler)

    Args:
        multiplier (float): constant multiplication value.

    """

    TYPE: ClassVar[str] = "multiply"
    _multiplier: Optional[float]

    def __init__(self, multiplier: Optional[float] = None):
        self._multiplier = multiplier

    def fit(self, transitions: List[Transition]) -> None:
        if self._multiplier is None:
            LOG.warning("Please initialize MultiplyRewardScaler manually.")

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        return self._multiplier * reward

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward / self._multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        return self._multiplier * reward

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"multiplier": self._multiplier}


class ClipRewardScaler(RewardScaler):
    r"""Reward clipping preprocessing.

    .. code-block:: python

        from tjuOfflineRL.preprocessing import ClipRewardScaler

        # clip rewards within [-1.0, 1.0]
        reward_scaler = ClipRewardScaler(low=-1.0, high=1.0)

        cql = CQL(reward_scaler=reward_scaler)

    Args:
        low (float): minimum value to clip.
        high (float): maximum value to clip.
        multiplier (float): constant multiplication value.

    """

    TYPE: ClassVar[str] = "clip"
    _low: Optional[float]
    _high: Optional[float]
    _multiplier: float

    def __init__(
        self,
        low: Optional[float] = None,
        high: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        self._low = low
        self._high = high
        self._multiplier = multiplier

    def fit(self, transitions: List[Transition]) -> None:
        if self._low is None and self._high is None:
            LOG.warning("Please initialize ClipRewardScaler manually.")

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        return self._multiplier * reward.clamp(self._low, self._high)

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward / self._multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        return self._multiplier * np.clip(reward, self._low, self._high)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "low": self._low,
            "high": self._high,
            "multiplier": self._multiplier,
        }


class MinMaxRewardScaler(RewardScaler):
    r"""Min-Max reward normalization preprocessing.

    .. math::

        r' = (r - \min(r)) / (\max(r) - \min(r))

    .. code-block:: python

        from tjuOfflineRL.algos import CQL

        cql = CQL(reward_scaler="min_max")

    You can also initialize with :class:`tjuOfflineRL.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from tjuOfflineRL.preprocessing import MinMaxRewardScaler

        # initialize with dataset
        scaler = MinMaxRewardScaler(dataset)

        # initialize manually
        scaler = MinMaxRewardScaler(minimum=0.0, maximum=10.0)

        cql = CQL(scaler=scaler)

    Args:
        dataset (tjuOfflineRL.dataset.MDPDataset): dataset object.
        minimum (float): minimum value.
        maximum (float): maximum value.
        multiplier (float): constant multiplication value.

    """
    TYPE: ClassVar[str] = "min_max"
    _minimum: Optional[float]
    _maximum: Optional[float]
    _multiplier: float

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        self._minimum = None
        self._maximum = None
        self._multiplier = multiplier
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif minimum is not None and maximum is not None:
            self._minimum = minimum
            self._maximum = maximum

    def fit(self, transitions: List[Transition]) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        rewards = [transition.reward for transition in transitions]

        self._minimum = float(np.min(rewards))
        self._maximum = float(np.max(rewards))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        base = self._maximum - self._minimum
        return self._multiplier * (reward - self._minimum) / base

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        base = self._maximum - self._minimum
        return reward * base / self._multiplier + self._minimum

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        base = self._maximum - self._minimum
        return self._multiplier * (reward - self._minimum) / base

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "minimum": self._minimum,
            "maximum": self._maximum,
            "multiplier": self._multiplier,
        }


class StandardRewardScaler(RewardScaler):
    r"""Reward standardization preprocessing.

    .. math::

        r' = (r - \mu) / \sigma

    .. code-block:: python

        from tjuOfflineRL.algos import CQL

        cql = CQL(reward_scaler="standard")

    You can also initialize with :class:`tjuOfflineRL.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from tjuOfflineRL.preprocessing import StandardRewardScaler

        # initialize with dataset
        scaler = StandardRewardScaler(dataset)

        # initialize manually
        scaler = StandardRewardScaler(mean=0.0, std=1.0)

        cql = CQL(scaler=scaler)

    Args:
        dataset (tjuOfflineRL.dataset.MDPDataset): dataset object.
        mean (float): mean value.
        std (float): standard deviation value.
        eps (float): constant value to avoid zero-division.
        multiplier (float): constant multiplication value

    """
    TYPE: ClassVar[str] = "standard"
    _mean: Optional[float]
    _std: Optional[float]
    _eps: float
    _multiplier: float

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        eps: float = 1e-3,
        multiplier: float = 1.0,
    ):
        self._mean = None
        self._std = None
        self._eps = eps
        self._multiplier = multiplier
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif mean is not None and std is not None:
            self._mean = mean
            self._std = std

    def fit(self, transitions: List[Transition]) -> None:
        if self._mean is not None and self._std is not None:
            return

        rewards = [transition.reward for transition in transitions]

        self._mean = float(np.mean(rewards))
        self._std = float(np.std(rewards))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        nonzero_std = self._std + self._eps
        return self._multiplier * (reward - self._mean) / nonzero_std

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        return reward * (self._std + self._eps) / self._multiplier + self._mean

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        nonzero_std = self._std + self._eps
        return self._multiplier * (reward - self._mean) / nonzero_std

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "mean": self._mean,
            "std": self._std,
            "eps": self._eps,
            "multiplier": self._multiplier,
        }


class ReturnBasedRewardScaler(RewardScaler):
    r"""Reward normalization preprocessing based on return scale.

    .. math::

        r' = r / (R_{max} - R_{min})

    .. code-block:: python

        from tjuOfflineRL.algos import CQL

        cql = CQL(reward_scaler="return")

    You can also initialize with :class:`tjuOfflineRL.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from tjuOfflineRL.preprocessing import ReturnBasedRewardScaler

        # initialize with dataset
        scaler = ReturnBasedRewardScaler(dataset)

        # initialize manually
        scaler = ReturnBasedRewardScaler(return_max=100.0, return_min=1.0)

        cql = CQL(scaler=scaler)

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        dataset (tjuOfflineRL.dataset.MDPDataset): dataset object.
        return_max (float): the maximum return value.
        return_min (float): standard deviation value.
        multiplier (float): constant multiplication value

    """
    TYPE: ClassVar[str] = "return"
    _return_max: Optional[float]
    _return_min: Optional[float]
    _multiplier: float

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        return_max: Optional[float] = None,
        return_min: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        self._return_max = None
        self._return_min = None
        self._multiplier = multiplier
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif return_max is not None and return_min is not None:
            self._return_max = return_max
            self._return_min = return_min

    def fit(self, transitions: List[Transition]) -> None:
        if self._return_max is not None and self._return_min is not None:
            return

        # collect start states
        start_transitions = set()
        for transition in transitions:
            # trace back to the start state
            curr_transition = transition
            while curr_transition.prev_transition:
                curr_transition = curr_transition.prev_transition
            start_transitions.add(curr_transition)

        # accumulate all rewards
        returns = []
        for start_transition in start_transitions:
            ret = 0.0
            curr_transition = start_transition
            while True:
                ret += curr_transition.reward
                if curr_transition.next_transition is None:
                    break
                curr_transition = curr_transition.next_transition
            returns.append(ret)

        self._return_max = float(np.max(returns))
        self._return_min = float(np.min(returns))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._return_max is not None and self._return_min is not None
        return self._multiplier * reward / (self._return_max - self._return_min)

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self._return_max is not None and self._return_min is not None
        return reward * (self._return_max + self._return_min) / self._multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self._return_max is not None and self._return_min is not None
        return self._multiplier * reward / (self._return_max - self._return_min)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "return_max": self._return_max,
            "return_min": self._return_min,
            "multiplier": self._multiplier,
        }


REWARD_SCALER_LIST: Dict[str, Type[RewardScaler]] = {}


def register_reward_scaler(cls: Type[RewardScaler]) -> None:
    """Registers reward scaler class.

    Args:
        cls: scaler class inheriting ``RewardScaler``.

    """
    is_registered = cls.TYPE in REWARD_SCALER_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    REWARD_SCALER_LIST[cls.TYPE] = cls


def create_reward_scaler(name: str, **kwargs: Any) -> RewardScaler:
    assert name in REWARD_SCALER_LIST, f"{name} seems not to be registered."
    reward_scaler = REWARD_SCALER_LIST[name](**kwargs)
    assert isinstance(reward_scaler, RewardScaler)
    return reward_scaler


register_reward_scaler(MultiplyRewardScaler)
register_reward_scaler(ClipRewardScaler)
register_reward_scaler(MinMaxRewardScaler)
register_reward_scaler(StandardRewardScaler)
register_reward_scaler(ReturnBasedRewardScaler)
