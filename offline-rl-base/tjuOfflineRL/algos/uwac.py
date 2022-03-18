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
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.uwac_impl import UWACImpl


class UWAC(AlgoBase):
    r"""UWAC is based on BEAR (Bootstrapping Error Accumulation Reduction algorithm).

    UWAC is a SAC-based data-driven deep reinforcement learning algorithm.

    BEAR constrains the support of the policy function within data distribution
    by minimizing Maximum Mean Discreptancy (MMD) between the policy function
    and the approximated beahvior policy function :math:`\pi_\beta(a|s)`
    which is optimized through L2 loss.

    UWAC contrins an extra uncertainty weight, on the critic loss.

    .. math::

        L(\beta) = \mathbb{E}_{s_t, a_t \sim D, a \sim
            \pi_\beta(\cdot|s_t)} [(a - a_t)^2]

    The policy objective is a combination of SAC's objective and MMD penalty.

    .. math::

        J(\phi) = J_{SAC}(\phi) - \mathbb{E}_{s_t \sim D} \alpha (
            \text{MMD}(\pi_\beta(\cdot|s_t), \pi_\phi(\cdot|s_t))
            - \epsilon)

    where MMD is computed as follows.

    .. math::

        \text{MMD}(x, y) = \frac{1}{N^2} \sum_{i, i'} k(x_i, x_{i'})
            - \frac{2}{NM} \sum_{i, j} k(x_i, y_j)
            + \frac{1}{M^2} \sum_{j, j'} k(y_j, y_{j'})

    where :math:`k(x, y)` is a gaussian kernel
    :math:`k(x, y) = \exp{((x - y)^2 / (2 \sigma^2))}`.

    :math:`\alpha` is also adjustable through dual gradient decsent where
    :math:`\alpha` becomes smaller if MMD is smaller than the threshold
    :math:`\epsilon`.

    References:
        * `Kumar et al., Stabilizing Off-Policy Q-Learning via Bootstrapping
          Error Reduction. <https://arxiv.org/abs/1906.00949>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for behavior policy
            function.
        temp_learning_rate (float): learning rate for temperature parameter.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the behavior policy.
        temp_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
        actor_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        imitator_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the behavior policy.
        q_func_factory (tjuOfflineRL.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as
            :math:`\epsilon`.
        lam (float): weight for critic ensemble.
        n_action_samples (int): the number of action samples to compute the
            best action.
        n_target_samples (int): the number of action samples to compute
            BCQ-like target value.
        n_mmd_action_samples (int): the number of action samples to compute MMD.
        mmd_kernel (str): MMD kernel function. The available options are
            ``['gaussian', 'laplacian']``.
        mmd_sigma (float): :math:`\sigma` for gaussian kernel in MMD
            calculation.
        vae_kl_weight (float): constant weight to scale KL term for behavior
            policy training.
        warmup_steps (int): the number of steps to warmup the policy
            function.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device iD or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The avaiable options are `['pixel', 'min_max', 'standard']`.
        action_scaler (tjuOfflineRL.preprocessing.ActionScaler or str):
            action preprocessor. The avaiable options are ``['min_max']``.
        reward_scaler (tjuOfflineRL.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (tjuOfflineRL.algos.torch.uwac_impl.UWACImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _temp_learning_rate: float
    _alpha_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _initial_temperature: float
    _initial_alpha: float
    _alpha_threshold: float
    _lam: float
    _n_action_samples: int
    _n_target_samples: int
    _n_mmd_action_samples: int
    _mmd_kernel: str
    _mmd_sigma: float
    _vae_kl_weight: float
    _warmup_steps: int
    _use_gpu: Optional[Device]
    _impl: Optional[UWACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        imitator_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        alpha_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 0.05,
        lam: float = 0.75,
        n_action_samples: int = 100,
        n_target_samples: int = 10,
        n_mmd_action_samples: int = 4,
        mmd_kernel: str = "laplacian",
        mmd_sigma: float = 20.0,
        vae_kl_weight: float = 0.5,
        warmup_steps: int = 40000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[UWACImpl] = None,
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
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._imitator_learning_rate = imitator_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._alpha_optim_factory = alpha_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._initial_temperature = initial_temperature
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._n_target_samples = n_target_samples
        self._n_mmd_action_samples = n_mmd_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma
        self._vae_kl_weight = vae_kl_weight
        self._warmup_steps = warmup_steps
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = UWACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            alpha_learning_rate=self._alpha_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            alpha_optim_factory=self._alpha_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            initial_alpha=self._initial_alpha,
            alpha_threshold=self._alpha_threshold,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            n_target_samples=self._n_target_samples,
            n_mmd_action_samples=self._n_mmd_action_samples,
            mmd_kernel=self._mmd_kernel,
            mmd_sigma=self._mmd_sigma,
            vae_kl_weight=self._vae_kl_weight,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        imitator_loss = self._impl.update_imitator(batch)
        metrics.update({"imitator_loss": imitator_loss})

        # lagrangian parameter update for SAC temperature
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        # lagrangian parameter update for MMD loss weight
        if self._alpha_learning_rate > 0:
            alpha_loss, alpha = self._impl.update_alpha(batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        if self._grad_step < self._warmup_steps:
            actor_loss = self._impl.warmup_actor(batch)
        else:
            actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        self._impl.update_actor_target()
        self._impl.update_critic_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
