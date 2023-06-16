from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ..algos import AlgoBase
from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import (
    ALGO_NOT_GIVEN_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from d3rlpy.dataset import TransitionMiniBatch
from ..dynamics import DynamicsBase, ProbabilisticEnsembleDynamics
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .torch.fqe_impl import DiscreteFQEImpl, FQEBaseImpl, FQEImpl


class _FQEBase(AlgoBase):

    _algo: Optional[AlgoBase]
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _target_update_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[FQEBaseImpl]

    def __init__(
        self,
        *,
        algo: Optional[AlgoBase] = None,
        learning_rate: float = 1e-4,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        target_update_interval: int = 100,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[FQEBaseImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._algo = algo
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._target_update_interval = target_update_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def save_policy(self, fname: str) -> None:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        self._algo.save_policy(fname)

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.predict(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action(x)

    def sample_action_dynamic(self, x: torch.Tensor, index: int) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action_dynamic(x, index)

    def sample_action_2pi(self, x: torch.Tensor, agent_offline) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action_2pi(x, agent_offline)

    def sample_action_reachability(self, x: torch.Tensor, offline_obs: np.array, total_step: int, explore_alpha: float) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action_reachability(x, offline_obs, total_step, explore_alpha)

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        next_actions = self._algo.predict(batch.next_observations)
        loss = self._impl.update(batch, next_actions)
        if self._grad_step % self._target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}


class FQE(_FQEBase):
    r"""Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """

    _impl: Optional[FQEImpl]

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = FQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


class DiscreteFQE(_FQEBase):
    r"""Fitted Q Evaluation for discrete action-space.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """

    _impl: Optional[DiscreteFQEImpl]

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteFQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=None,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
