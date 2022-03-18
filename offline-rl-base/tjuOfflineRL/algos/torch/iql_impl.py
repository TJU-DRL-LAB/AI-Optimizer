from typing import Optional, Sequence

import numpy as np
import torch

from ...gpu import Device
from ...models.builders import (
    create_squashed_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import MeanQFunctionFactory
from ...models.torch import SquashedNormalPolicy, ValueFunction, squash_action
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .ddpg_impl import DDPGBaseImpl


class IQLImpl(DDPGBaseImpl):
    _policy: Optional[SquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
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
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=MeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_value_function(
            self._observation_shape, self._value_encoder_factory
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            return self._value_func(batch.next_observations)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy

        dist = self._policy.dist(batch.observations)

        # unnormalize action via inverse tanh function
        clipped_actions = batch.actions.clamp(-0.999999, 0.999999)
        unnormalized_act_t = torch.atanh(clipped_actions)

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_act_t)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._critic_optim.step()

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()
