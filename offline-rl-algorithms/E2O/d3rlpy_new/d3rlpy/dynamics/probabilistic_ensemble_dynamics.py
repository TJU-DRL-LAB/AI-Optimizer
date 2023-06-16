from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from .base import DynamicsBase
from .torch.probabilistic_ensemble_dynamics_impl import (
    ProbabilisticEnsembleDynamicsImpl,
)


class ProbabilisticEnsembleDynamics(DynamicsBase):
    r"""Probabilistic ensemble dynamics.

    The ensemble dynamics model consists of :math:`N` probablistic models
    :math:`\{T_{\theta_i}\}_{i=1}^N`.
    At each epoch, new transitions are generated via randomly picked dynamics
    model :math:`T_\theta`.

    .. math::

        s_{t+1}, r_{t+1} \sim T_\theta(s_t, a_t)

    where :math:`s_t \sim D` for the first step, otherwise :math:`s_t` is the
    previous generated observation, and :math:`a_t \sim \pi(\cdot|s_t)`.

    Note:
        Currently, ``ProbabilisticEnsembleDynamics`` only supports vector
        observations.

    References:
        * `Yu et al., MOPO: Model-based Offline Policy Optimization.
          <https://arxiv.org/abs/2005.13239>`_

    Args:
        learning_rate (float): learning rate for dynamics model.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_ensembles (int): the number of dynamics model for ensemble.
        variance_type (str): variance calculation type. The available options
            are ``['max', 'data']``.
        discrete_action (bool): flag to take discrete actions.
        scaler (d3rlpy.preprocessing.scalers.Scaler or str): preprocessor.
            The available options are ``['pixel', 'min_max', 'standard']``.
        action_scaler (d3rlpy.preprocessing.Actionscalers or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        impl (d3rlpy.dynamics.torch.ProbabilisticEnsembleDynamicsImpl):
            dynamics implementation.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _n_ensembles: int
    _variance_type: str
    _discrete_action: bool
    _use_gpu: Optional[Device]
    _impl: Optional[ProbabilisticEnsembleDynamicsImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(weight_decay=1e-4),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        n_ensembles: int = 5,
        variance_type: str = "max",
        discrete_action: bool = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        use_gpu: UseGPUArg = False,
        impl: Optional[ProbabilisticEnsembleDynamicsImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._n_ensembles = n_ensembles
        self._variance_type = variance_type
        self._discrete_action = discrete_action
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = ProbabilisticEnsembleDynamicsImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            n_ensembles=self._n_ensembles,
            variance_type=self._variance_type,
            discrete_action=self._discrete_action,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.BOTH
