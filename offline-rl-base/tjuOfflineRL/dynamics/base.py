from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..argument_utility import ActionScalerArg, RewardScalerArg, ScalerArg
from ..base import ImplBase, LearnableBase
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class DynamicsImplBase(ImplBase):
    @abstractmethod
    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        indices: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class DynamicsBase(LearnableBase):

    _impl: Optional[DynamicsImplBase]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        scaler: ScalerArg,
        action_scaler: ActionScalerArg,
        reward_scaler: RewardScalerArg,
        kwargs: Dict[str, Any],
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=1.0,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._impl = None

    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_variance: bool = False,
        indices: Optional[np.ndarray] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Returns predicted observation and reward.

        Args:
            x: observation
            action: action
            with_variance: flag to return prediction variance.
            indices: index of ensemble model to return.

        Returns:
            tuple of predicted observation and reward. If ``with_variance`` is
            ``True``, the prediction variance will be added as the 3rd element.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        observations, rewards, variances = self._impl.predict(
            x,
            action,
            indices,
        )
        if with_variance:
            return observations, rewards, variances
        return observations, rewards
