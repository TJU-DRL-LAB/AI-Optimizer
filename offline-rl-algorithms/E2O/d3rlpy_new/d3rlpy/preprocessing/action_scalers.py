from typing import Any, ClassVar, Dict, List, Optional, Type

import gym
import numpy as np
import torch

from d3rlpy.dataset import MDPDataset, Transition
from d3rlpy.decorators import pretty_repr


@pretty_repr
class ActionScaler:
    TYPE: ClassVar[str] = "none"

    def fit(self, transitions: List[Transition]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            transitions: a list of transition objects.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        """Returns processed action.

        Args:
            action: action vector.

        Returns:
            processed action.

        """
        raise NotImplementedError

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed action.

        Args:
            action: action vector.

        Returns:
            reversely transformed action.

        """
        raise NotImplementedError

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        """Returns reversely transformed action in numpy array.

        Args:
            action: action vector.

        Returns:
            reversely transformed action.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns action scaler type.

        Returns:
            action scaler type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns action scaler params.

        Args:
            deep: flag to deepcopy parameters.

        Returns:
            action scaler parameters.

        """
        raise NotImplementedError


class MinMaxActionScaler(ActionScaler):
    r"""Min-Max normalization action preprocessing.

    Actions will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxActionScaler
        cql = CQL(action_scaler='min_max')

        # scaler is initialized from the given transitions
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxActionScaler

        # initialize with dataset
        scaler = MinMaxActionScaler(dataset)

        # initialize manually
        minimum = actions.min(axis=0)
        maximum = actions.max(axis=0)
        action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)

        cql = CQL(action_scaler=action_scaler)

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
            action = np.asarray(transition.action)
            if i == 0:
                minimum = action
                maximum = action
            else:
                minimum = np.minimum(minimum, action)
                maximum = np.maximum(maximum, action)

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.action_space, gym.spaces.Box)
        shape = env.action_space.shape
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=action.device
        )
        # transform action into [-1.0, 1.0]
        return ((action - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=action.device
        )
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        return {"minimum": minimum, "maximum": maximum}


ACTION_SCALER_LIST: Dict[str, Type[ActionScaler]] = {}


def register_action_scaler(cls: Type[ActionScaler]) -> None:
    """Registers action scaler class.

    Args:
        cls: action scaler class inheriting ``ActionScaler``.

    """
    is_registered = cls.TYPE in ACTION_SCALER_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    ACTION_SCALER_LIST[cls.TYPE] = cls


def create_action_scaler(name: str, **kwargs: Any) -> ActionScaler:
    """Returns registered action scaler object.

    Args:
        name: regsitered scaler type name.
        kwargs: scaler arguments.

    Returns:
        scaler object.

    """
    assert name in ACTION_SCALER_LIST, f"{name} seems not to be registered."
    scaler = ACTION_SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, ActionScaler)
    return scaler


register_action_scaler(MinMaxActionScaler)
