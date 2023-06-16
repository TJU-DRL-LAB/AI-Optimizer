import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_parameter,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    Parameter,
    SquashedNormalPolicy,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .ddpg_impl import DDPGBaseImpl


class E2OImpl(DDPGBaseImpl):

    _policy: Optional[SquashedNormalPolicy]
    _targ_policy: Optional[SquashedNormalPolicy]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]

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
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._temp_learning_rate = temp_learning_rate
        self._temp_optim_factory = temp_optim_factory
        self._initial_temperature = initial_temperature

        self._log_temp = None
        self._temp_optim = None

    def build(self) -> None:
        self._build_temperature()
        super().build()
        self._build_temperature_optim()

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        return (entropy - q_t).mean()

    @train_api
    @torch_api()
    def update_temp(
        self, batch: TorchMiniBatch
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob = self._policy.sample_with_log_prob(batch.observations)
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func.compute_target_weighted(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        q_stds = []
        for i in range(self._n_critics):
            q_stds.append(self._targ_q_func.compute_q_std(
                batch.next_observations,
                self._policy.sample(batch.next_observations),
            ).item())

        q_std = torch.tensor(
            np.mean(q_stds), dtype=torch.float32, device=batch.device
        )

        temperature = 10

        ratio = torch.sigmoid(-q_std * temperature) + 0.5

        loss = ratio.item() * self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def boltzmann(self, x, temperature):
        exponent = np.true_divide(x - np.max(x), temperature)
        return np.exp(exponent) / np.sum(np.exp(exponent))

    def sample_action_dynamic(self, x: torch.Tensor, index: int) -> torch.Tensor:
        assert self._policy is not None

        actions = []

        # SUNRISE
        q_means = []
        q_stds = []
        for i in range(self._n_critics):
            action = self._policy.sample(x)
            actions.append(action)
            q_means.append(self._q_func(x.unsqueeze(0), action.unsqueeze(0), reduction="mean").item())
            q_stds.append(self._q_func(x.unsqueeze(0), action.unsqueeze(0), reduction="std").item())
        lamda = 10
        index = torch.argmax(torch.from_numpy(np.array(q_means) + lamda * np.array(q_stds))).item()

        # PEX
        # q_qstd = np.array(q_means) + lamda * np.array(q_stds)
        # pw = self.boltzmann(q_qstd, 1/3)
        # index = torch.distributions.Categorical(torch.from_numpy(pw)).sample().item()

        # Bootstrapped DQN
        # q_values = []
        # for i in range(self._n_critics):
        #     action = self._policy.sample(x)
        #     actions.append(action)
        #     q_values.append(self._q_func.compute_single_q(x.unsqueeze(0), action.unsqueeze(0), index).item())
        # index = torch.argmax(torch.from_numpy(np.array(q_values))).item()

        return actions[index].cpu().detach().numpy()

        # OAC
        # beta_UB = 4.66
        # delta = 23.53
        # pre_tanh_mu_T, std = self._policy.mu_std(x)
        # assert len(list(pre_tanh_mu_T.shape)) == 1, pre_tanh_mu_T
        # assert len(list(std.shape)) == 1
        # pre_tanh_mu_T.requires_grad_()
        # tanh_mu_T = torch.tanh(pre_tanh_mu_T)
        # mu_Q = self._q_func(x.unsqueeze(0), tanh_mu_T.unsqueeze(0), reduction="mean")
        # sigma_Q = self._q_func(x.unsqueeze(0), tanh_mu_T.unsqueeze(0), reduction="std")
        # Q_UB = mu_Q + beta_UB * sigma_Q
        # grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
        # grad = grad[0]
        # assert grad is not None
        # assert pre_tanh_mu_T.shape == grad.shape
        # Sigma_T = torch.pow(std, 2)
        # denom = torch.sqrt(
        #     torch.sum(
        #         torch.mul(torch.pow(grad, 2), Sigma_T)
        #     )
        # ) + 10e-6
        # mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom
        # assert mu_C.shape == pre_tanh_mu_T.shape
        # mu_E = pre_tanh_mu_T + mu_C
        # assert mu_E.shape == std.shape
        # dist = torch.distributions.Normal(mu_E, std)
        # z = dist.sample().detach()
        # ac = torch.tanh(z)
        # return ac.cpu().detach().numpy()
