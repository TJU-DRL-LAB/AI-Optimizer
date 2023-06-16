from typing import Any, Dict, Optional, Sequence

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
from d3rlpy.dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.crr_impl import CRRImpl


class CRR(AlgoBase):
    r"""Critic Reguralized Regression algorithm.

    CRR is a simple offline RL method similar to AWAC.

    The policy is trained as a supervised regression.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t \sim D}
            [\log \pi_\phi(a_t|s_t) f(Q_\theta, \pi_\phi, s_t, a_t)]

    where :math:`f` is a filter function which has several options. The first
    option is ``binary`` function.

    .. math::

        f := \mathbb{1} [A_\theta(s, a) > 0]

    The other is ``exp`` function.

    .. math::

        f := \exp(A(s, a) / \beta)

    The :math:`A(s, a)` is an average function which also has several options.
    The first option is ``mean``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \frac{1}{m} \sum^m_j Q(s, a_j)

    The other one is ``max``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \max^m_j Q(s, a_j)

    where :math:`a_j \sim \pi_\phi(s)`.

    In evaluation, the action is determined by Critic Weighted Policy (CWP).
    In CWP, the several actions are sampled from the policy function, and the
    final action is re-sampled from the estimated action-value distribution.

    References:
        * `Wang et al., Critic Reguralized Regression.
          <https://arxiv.org/abs/2006.15134>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        beta (float): temperature value defined as :math:`\beta` above.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A(s, a)` and for CWP.
        advantage_type (str): advantage function type. The available options
            are ``['mean', 'max']``.
        weight_type (str): filter function type. The available options
            are ``['binary', 'exp']``.
        max_weight (float): maximum weight for cross-entropy loss.
        n_critics (int): the number of Q functions for ensemble.
        target_update_type (str): target update type. The available options are
            ``['hard', 'soft']``.
        tau (float): target network synchronization coefficiency used with
            ``soft`` target update.
        update_actor_interval (int): interval to update policy function used
            with ``hard`` target update.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.crr_impl.CRRImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _beta: float
    _n_action_samples: int
    _advantage_type: str
    _weight_type: str
    _max_weight: float
    _n_critics: int
    _target_update_type: str
    _tau: float
    _target_update_interval: int
    _update_actor_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[CRRImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        beta: float = 1.0,
        n_action_samples: int = 4,
        advantage_type: str = "mean",
        weight_type: str = "exp",
        max_weight: float = 20.0,
        n_critics: int = 1,
        target_update_type: str = "hard",
        tau: float = 5e-3,
        target_update_interval: int = 100,
        update_actor_interval: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[CRRImpl] = None,
        **kwargs: Any,
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
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._beta = beta
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._max_weight = max_weight
        self._n_critics = n_critics
        self._target_update_type = target_update_type
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._update_actor_interval = update_actor_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CRRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            beta=self._beta,
            n_action_samples=self._n_action_samples,
            advantage_type=self._advantage_type,
            weight_type=self._weight_type,
            max_weight=self._max_weight,
            n_critics=self._n_critics,
            tau=self._tau,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        critic_loss = self._impl.update_critic(batch)
        actor_loss = self._impl.update_actor(batch)

        if self._target_update_type == "hard":
            if self._grad_step % self._target_update_interval == 0:
                self._impl.sync_critic_target()
                self._impl.sync_actor_target()
        elif self._target_update_type == "soft":
            self._impl.update_critic_target()
            self._impl.update_actor_target()
        else:
            raise ValueError(
                f"invalid target_update_type: {self._target_update_type}"
            )

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
