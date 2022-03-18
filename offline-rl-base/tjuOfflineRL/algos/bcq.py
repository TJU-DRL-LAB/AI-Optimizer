from typing import Any, Dict, List, Optional, Sequence, Union

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
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.bcq_impl import BCQImpl, DiscreteBCQImpl


class BCQ(AlgoBase):
    r"""Batch-Constrained Q-learning algorithm.

    BCQ is the very first practical data-driven deep reinforcement learning
    lgorithm.
    The major difference from DDPG is that the policy function is represented
    as combination of conditional VAE and perturbation function in order to
    remedy extrapolation error emerging from target value estimation.

    The encoder and the decoder of the conditional VAE is represented as
    :math:`E_\omega` and :math:`D_\omega` respectively.

    .. math::

        L(\omega) = E_{s_t, a_t \sim D} [(a - \tilde{a})^2
            + D_{KL}(N(\mu, \sigma)|N(0, 1))]

    where :math:`\mu, \sigma = E_\omega(s_t, a_t)`,
    :math:`\tilde{a} = D_\omega(s_t, z)` and :math:`z \sim N(\mu, \sigma)`.

    The policy function is represented as a residual function
    with the VAE and the perturbation function represented as
    :math:`\xi_\phi (s, a)`.

    .. math::

        \pi(s, a) = a + \Phi \xi_\phi (s, a)

    where :math:`a = D_\omega (s, z)`, :math:`z \sim N(0, 0.5)` and
    :math:`\Phi` is a perturbation scale designated by `action_flexibility`.
    Although the policy is learned closely to data distribution, the
    perturbation function can lead to more rewarded states.

    BCQ also leverages twin Q functions and computes weighted average over
    maximum values and minimum values.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(y - Q_{\theta_i}(s_t, a_t))^2]

    .. math::

        y = r_{t+1} + \gamma \max_{a_i} [
            \lambda \min_j Q_{\theta_j'}(s_{t+1}, a_i)
            + (1 - \lambda) \max_j Q_{\theta_j'}(s_{t+1}, a_i)]

    where :math:`\{a_i \sim D(s_{t+1}, z), z \sim N(0, 0.5)\}_{i=1}^n`.
    The number of sampled actions is designated with `n_action_samples`.

    Finally, the perturbation function is trained just like DDPG's policy
    function.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D, a_t \sim D_\omega(s_t, z),
                              z \sim N(0, 0.5)}
            [Q_{\theta_1} (s_t, \pi(s_t, a_t))]

    At inference time, action candidates are sampled as many as
    `n_action_samples`, and the action with highest value estimation is taken.

    .. math::

        \pi'(s) = \text{argmax}_{\pi(s, a_i)} Q_{\theta_1} (s, \pi(s, a_i))

    Note:
        The greedy action is not deterministic because the action candidates
        are always randomly sampled. This might affect `save_policy` method and
        the performance at production.

    References:
        * `Fujimoto et al., Off-Policy Deep Reinforcement Learning without
          Exploration. <https://arxiv.org/abs/1812.02900>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        imitator_encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory for the conditional VAE.
        q_func_factory (tjuOfflineRL.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        action_flexibility (float): output scale of perturbation function
            represented as :math:`\Phi`.
        rl_start_step (int): step to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        beta (float): KL reguralization term for Conditional VAE.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (tjuOfflineRL.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (tjuOfflineRL.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (tjuOfflineRL.algos.torch.bcq_impl.BCQImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _rl_start_step: int
    _beta: float
    _use_gpu: Optional[Device]
    _impl: Optional[BCQImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        imitator_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        lam: float = 0.75,
        n_action_samples: int = 100,
        action_flexibility: float = 0.05,
        rl_start_step: int = 0,
        beta: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[BCQImpl] = None,
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
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_actor_interval = update_actor_interval
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._rl_start_step = rl_start_step
        self._beta = beta
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            action_flexibility=self._action_flexibility,
            beta=self._beta,
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

        if self._grad_step >= self._rl_start_step:
            critic_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})

            if self._grad_step % self._update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch)
                metrics.update({"actor_loss": actor_loss})
                self._impl.update_actor_target()
                self._impl.update_critic_target()

        return metrics

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """BCQ does not support sampling action."""
        raise NotImplementedError("BCQ does not support sampling action.")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


class DiscreteBCQ(AlgoBase):
    r"""Discrete version of Batch-Constrained Q-learning algorithm.

    Discrete version takes theories from the continuous version, but the
    algorithm is much simpler than that.
    The imitation function :math:`G_\omega(a|s)` is trained as supervised
    learning just like Behavior Cloning.

    .. math::

        L(\omega) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log G_\omega(a|s_t)]

    With this imitation function, the greedy policy is defined as follows.

    .. math::

        \pi(s_t) = \text{argmax}_{a|G_\omega(a|s_t)
                / \max_{\tilde{a}} G_\omega(\tilde{a}|s_t) > \tau}
            Q_\theta (s_t, a)

    which eliminates actions with probabilities :math:`\tau` times smaller
    than the maximum one.

    Finally, the loss function is computed in Double DQN style with the above
    constrained policy.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\theta'}(s_{t+1}, \pi(s_{t+1}))
            - Q_\theta(s_t, a_t))^2]

    References:
        * `Fujimoto et al., Off-Policy Deep Reinforcement Learning without
          Exploration. <https://arxiv.org/abs/1812.02900>`_
        * `Fujimoto et al., Benchmarking Batch Deep Reinforcement Learning
          Algorithms. <https://arxiv.org/abs/1910.01708>`_

    Args:
        learning_rate (float): learning rate.
        optim_factory (tjuOfflineRL.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (tjuOfflineRL.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (tjuOfflineRL.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        action_flexibility (float): probability threshold represented as
            :math:`\tau`.
        beta (float): reguralization term for imitation function.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or tjuOfflineRL.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (tjuOfflineRL.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        reward_scaler (tjuOfflineRL.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (tjuOfflineRL.algos.torch.bcq_impl.DiscreteBCQImpl):
            algorithm implementation.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _action_flexibility: float
    _beta: float
    _target_update_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[DiscreteBCQImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        action_flexibility: float = 0.3,
        beta: float = 0.5,
        target_update_interval: int = 8000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[DiscreteBCQImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._action_flexibility = action_flexibility
        self._beta = beta
        self._target_update_interval = target_update_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteBCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            action_flexibility=self._action_flexibility,
            beta=self._beta,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        if self._grad_step % self._target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
