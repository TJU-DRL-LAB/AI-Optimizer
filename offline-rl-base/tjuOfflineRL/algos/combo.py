from typing import Any, Dict, List, Optional, Sequence

import numpy as np

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
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Transition, TransitionMiniBatch
from ..dynamics import DynamicsBase
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.combo_impl import COMBOImpl
from .utility import ModelBaseMixin


class COMBO(ModelBaseMixin, AlgoBase):
    r"""Conservative Offline Model-Based Optimization.

    COMBO is a model-based RL approach for offline policy optimization.
    COMBO is similar to MOPO, but it also leverages conservative loss proposed
    in CQL.

    .. math::

        L(\theta_i) = \mathbb{E}_{s \sim d_M}
            \big[\log{\sum_a \exp{Q_{\theta_i}(s_t, a)}}\big]
             - \mathbb{E}_{s, a \sim D} \big[Q_{\theta_i}(s, a)\big]
            + L_\mathrm{SAC}(\theta_i)

    Note:
        Currently, COMBO only supports vector observations.

    References:
        * `Yu et al., COMBO: Conservative Offline Model-Based Policy
          Optimization. <https://arxiv.org/abs/2102.08363>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (tjuOfflineRL.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        dynamics (tjuOfflineRL.dynamics.DynamicsBase): dynamics object.
        rollout_interval (int): the number of steps before rollout.
        rollout_horizon (int): the rollout step length.
        rollout_batch_size (int): the number of initial transitions for
            rollout.
        real_ratio (float): the real of dataset samples in a mini-batch.
        generated_maxlen (int): the maximum number of generated samples.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (tjuOfflineRL.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (tjuOfflineRL.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (tjuOfflineRL.algos.torch.combo_impl.COMBOImpl):
            algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _initial_temperature: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _dynamics: Optional[DynamicsBase]
    _rollout_interval: int
    _rollout_horizon: int
    _rollout_batch_size: int
    _use_gpu: Optional[Device]
    _impl: Optional[COMBOImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        conservative_weight: float = 1.0,
        n_action_samples: int = 10,
        soft_q_backup: bool = False,
        dynamics: Optional[DynamicsBase] = None,
        rollout_interval: int = 1000,
        rollout_horizon: int = 5,
        rollout_batch_size: int = 50000,
        real_ratio: float = 0.5,
        generated_maxlen: int = 50000 * 5 * 5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[COMBOImpl] = None,
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
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_actor_interval = update_actor_interval
        self._initial_temperature = initial_temperature
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup
        self._dynamics = dynamics
        self._rollout_interval = rollout_interval
        self._rollout_horizon = rollout_horizon
        self._rollout_batch_size = rollout_batch_size
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = COMBOImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            conservative_weight=self._conservative_weight,
            n_action_samples=self._n_action_samples,
            real_ratio=self._real_ratio,
            soft_q_backup=self._soft_q_backup,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        # delayed policy update
        if self._grad_step % self._update_actor_interval == 0:
            actor_loss = self._impl.update_actor(batch)
            metrics.update({"actor_loss": actor_loss})

            # lagrangian parameter update for SAC temperature
            if self._temp_learning_rate > 0:
                temp_loss, temp = self._impl.update_temp(batch)
                metrics.update({"temp_loss": temp_loss, "temp": temp})

            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0

    def _sample_initial_transitions(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        # uniformly sample transitions
        n_transitions = self._rollout_batch_size
        indices = np.random.randint(len(transitions), size=n_transitions)
        return [transitions[i] for i in indices]

    def _get_rollout_horizon(self) -> int:
        return self._rollout_horizon
