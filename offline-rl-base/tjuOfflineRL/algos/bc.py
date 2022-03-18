from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from .base import AlgoBase
from .torch.bc_impl import BCBaseImpl, BCImpl, DiscreteBCImpl


class _BCBase(AlgoBase):
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _impl: Optional[BCBaseImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[BCBaseImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=1.0,
            scaler=scaler,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update_imitator(batch.observations, batch.actions)
        return {"loss": loss}

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> np.ndarray:
        """value prediction is not supported by BC algorithms."""
        raise NotImplementedError("BC does not support value estimation.")

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> None:
        """sampling action is not supported by BC algorithm."""
        raise NotImplementedError("BC does not support sampling action.")


class BC(_BCBase):
    r"""Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): learing rate.
        optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        policy_type (str): the policy type. The available options are
            ``['deterministic', 'stochastic']``.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (tjuOfflineRL.preprocessing.ActionScaler or str):
            action scaler. The available options are ``['min_max']``.
        impl (tjuOfflineRL.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.

    """

    _policy_type: str
    _impl: Optional[BCImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        policy_type: str = "deterministic",
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[BCBaseImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            impl=impl,
            **kwargs,
        )
        self._policy_type = policy_type

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            policy_type=self._policy_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


class DiscreteBC(_BCBase):
    r"""Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log \pi_\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        learning_rate (float): learing rate.
        optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        beta (float): reguralization factor.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        impl (tjuOfflineRL.algos.torch.bc_impl.DiscreteBCImpl):
            implemenation of the algorithm.

    """

    _beta: float
    _impl: Optional[DiscreteBCImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        beta: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        impl: Optional[DiscreteBCImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            use_gpu=use_gpu,
            scaler=scaler,
            impl=impl,
            **kwargs,
        )
        self._beta = beta

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            beta=self._beta,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
