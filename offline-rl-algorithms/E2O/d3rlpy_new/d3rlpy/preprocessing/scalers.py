from typing import Any, ClassVar, Dict, List, Optional, Type

import gym
import numpy as np
import torch
import re

from d3rlpy.dataset import MDPDataset, Transition
from ..decorators import pretty_repr


@pretty_repr
class Scaler:

    TYPE: ClassVar[str] = "none"

    def fit(self, transitions: List[Transition]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            transitions: list of transitions.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns processed observations.

        Args:
            x: observation.

        Returns:
            processed observation.

        """
        raise NotImplementedError

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observations.

        Args:
            x: observation.

        Returns:
            reversely transformed observation.

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


class PixelScaler(Scaler):
    """Pixel normalization preprocessing.

    .. math::

        x' = x / 255

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with PixelScaler
        cql = CQL(scaler='pixel')

        cql.fit(dataset.episodes)

    """

    TYPE: ClassVar[str] = "pixel"

    def fit(self, transitions: List[Transition]) -> None:
        pass

    def fit_with_env(self, env: gym.Env) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 255.0).long()

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {}


class MinMaxScaler(Scaler):
    r"""Min-Max normalization preprocessing.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxScaler
        cql = CQL(scaler='min_max')

        # scaler is initialized from the given transitions
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxScaler

        # initialize with dataset
        scaler = MinMaxScaler(dataset)

        # initialize manually
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

        cql = CQL(scaler=scaler)

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.
        min (numpy.ndarray): minimum values at each entry.
        max (numpy.ndarray): maximum values at each entry.

    """

    TYPE: ClassVar[str] = "min_max"
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, transitions: List[Transition]) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        for i, transition in enumerate(transitions):
            observation = np.asarray(transition.observation)
            if i == 0:
                minimum = observation
                maximum = observation
            else:
                minimum = np.minimum(minimum, observation)
                maximum = np.maximum(maximum, observation)

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.observation_space, gym.spaces.Box)
        if re.match(r"Hopper|HalfCheetah|Walker|Ant|Pen|Door|maze", env.spec.id):
            observation_shape = env.observation_space.shape
        else:
            observation_shape = env.observation_space.spaces['state'].shape
        shape = observation_shape
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return (x - minimum) / (maximum - minimum)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return ((maximum - minimum) * x) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        return {"maximum": maximum, "minimum": minimum}


class StandardScaler(Scaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with StandardScaler
        cql = CQL(scaler='standard')

        # scaler is initialized from the given episodes
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import StandardScaler

        # initialize with dataset
        scaler = StandardScaler(dataset)

        # initialize manually
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        scaler = StandardScaler(mean=mean, std=std)

        cql = CQL(scaler=scaler)

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.
        mean (numpy.ndarray): mean values at each entry.
        std (numpy.ndarray): standard deviation at each entry.
        eps (float): small constant value to avoid zero-division.

    """

    TYPE = "standard"
    _mean: Optional[np.ndarray]
    _std: Optional[np.ndarray]
    _eps: float

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        eps: float = 1e-3,
    ):
        self._mean = None
        self._std = None
        self._eps = eps
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif mean is not None and std is not None:
            self._mean = np.asarray(mean)
            self._std = np.asarray(std)

    def fit(self, transitions: List[Transition]) -> None:
        if self._mean is not None and self._std is not None:
            return

        # compute mean
        total_sum = np.zeros(transitions[0].get_observation_shape())
        total_count = 0
        for transition in transitions:
            total_sum += np.asarray(transition.observation)
            total_count += 1
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(transitions[0].get_observation_shape())
        expanded_mean = mean.reshape(mean.shape)
        for transition in transitions:
            observation = np.asarray(transition.observation)
            total_sqsum += (observation - expanded_mean) ** 2
        std = np.sqrt(total_sqsum / total_count)

        self._mean = mean.reshape((1,) + mean.shape)
        self._std = std.reshape((1,) + std.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._mean is not None and self._std is not None:
            return
        raise NotImplementedError(
            "standard scaler does not support fit_with_env."
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return (x - mean) / (std + self._eps)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return ((std + self._eps) * x) + mean

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._mean is not None:
            mean = self._mean.copy() if deep else self._mean
        else:
            mean = None

        if self._std is not None:
            std = self._std.copy() if deep else self._std
        else:
            std = None

        return {"mean": mean, "std": std, "eps": self._eps}


SCALER_LIST: Dict[str, Type[Scaler]] = {}


def register_scaler(cls: Type[Scaler]) -> None:
    """Registers scaler class.

    Args:
        cls: scaler class inheriting ``Scaler``.

    """
    is_registered = cls.TYPE in SCALER_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    SCALER_LIST[cls.TYPE] = cls


def create_scaler(name: str, **kwargs: Any) -> Scaler:
    """Returns registered scaler object.

    Args:
        name: regsitered scaler type name.
        kwargs: scaler arguments.

    Returns:
        scaler object.

    """
    assert name in SCALER_LIST, f"{name} seems not to be registered."
    scaler = SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, Scaler)
    return scaler


register_scaler(PixelScaler)
register_scaler(MinMaxScaler)
register_scaler(StandardScaler)
