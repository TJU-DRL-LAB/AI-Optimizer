# pylint: disable=too-many-ancestors

from typing import Optional, Sequence

import torch

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from .cql_impl import CQLImpl


class COMBOImpl(CQLImpl):

    _real_ratio: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        conservative_weight: float,
        n_action_samples: int,
        real_ratio: float,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            alpha_learning_rate=0.0,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            alpha_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            initial_alpha=1.0,
            alpha_threshold=0.0,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._real_ratio = real_ratio

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # split batch
        fake_obs_t = obs_t[int(obs_t.shape[0] * self._real_ratio) :]
        fake_obs_tp1 = obs_tp1[int(obs_tp1.shape[0] * self._real_ratio) :]
        real_obs_t = obs_t[: int(obs_t.shape[0] * self._real_ratio)]
        real_act_t = act_t[: int(act_t.shape[0] * self._real_ratio)]

        # compute conservative loss only with generated transitions
        random_values = self._compute_random_is_values(fake_obs_t)
        policy_values_t = self._compute_policy_is_values(fake_obs_t, fake_obs_t)
        policy_values_tp1 = self._compute_policy_is_values(
            fake_obs_tp1, fake_obs_t
        )

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for real data actions
        data_values = self._q_func(real_obs_t, real_act_t, "none")

        loss = logsumexp.sum(dim=0).mean() - data_values.sum(dim=0).mean()

        return loss
