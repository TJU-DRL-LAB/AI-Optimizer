from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from ..argument_utility import ActionScalerArg
from ..constants import ActionSpace
from .base import AlgoBase


class RandomPolicy(AlgoBase):
    r"""Random Policy for continuous control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.

    Args:
        distribution (str): random distribution. The available options are
            ``['uniform', 'normal']``.
        normal_std (float): standard deviation of the normal distribution. This
            is only used when ``distribution='normal'``.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.

    """

    _distribution: str
    _normal_std: float
    _action_size: int

    def __init__(
        self,
        *,
        distribution: str = "uniform",
        normal_std: float = 1.0,
        action_scaler: ActionScalerArg = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=1,
            n_frames=1,
            n_steps=1,
            gamma=0.0,
            scaler=None,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self._distribution = distribution
        self._normal_std = normal_std
        self._action_size = 1
        self._impl = None

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.sample_action(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        x = np.asarray(x)
        action_shape = (x.shape[0], self._action_size)

        if self._distribution == "uniform":
            action = np.random.uniform(-1.0, 1.0, size=action_shape)
        elif self._distribution == "normal":
            action = np.random.normal(0.0, self._normal_std, size=action_shape)
        else:
            raise ValueError(f"invalid distribution type: {self._distribution}")

        action = np.clip(action, -1.0, 1.0)

        if self._action_scaler:
            action = self._action_scaler.reverse_transform_numpy(action)

        return action

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


class DiscreteRandomPolicy(AlgoBase):
    r"""Random Policy for discrete control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.

    """

    _action_size: int

    def __init__(self, **kwargs: Any):
        super().__init__(
            batch_size=1,
            n_frames=1,
            n_steps=1,
            gamma=0.0,
            scaler=None,
            action_scaler=None,
            kwargs=kwargs,
        )
        self._action_size = 1
        self._impl = None

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.sample_action(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        x = np.asarray(x)
        return np.random.randint(self._action_size, size=x.shape[0])

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
