# pylint: disable=too-many-ancestors

from typing import Optional, Sequence

import torch
import numpy as np

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch
from .td3_impl import TD3Impl


class TD3PlusBCImpl(TD3Impl):

    _alpha: float

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
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
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
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha
        self.time_step = 1

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]

        # TD3 + BC
        # lam = self._alpha / (q_t.abs().mean()).detach()
        # return lam * -q_t.mean() + ((batch.actions - action) ** 2).mean()

        # TD3 + AdaBC
        # lam_max = 0.3
        # lam_ada = torch.from_numpy(np.array(lam_max - self.time_step / 250000 * lam_max))
        # self.time_step += 1
        # return -q_t.mean() + lam_ada * ((batch.actions - action) ** 2).mean()

        # Combination
        lam = self._alpha / (q_t.abs().mean()).detach()
        lam_max = 0.3
        lam_ada = torch.from_numpy(np.array(lam_max - self.time_step / 250000 * lam_max))
        self.time_step += 1
        return lam * -q_t.mean() + lam_ada * ((batch.actions - action) ** 2).mean()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func.compute_target_weighted(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    # def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
    #     assert self._critic_optim is not None
    #     self._critic_optim.zero_grad()
    #
    #     q_tpn = self.compute_target(batch)
    #     q_stds = []
    #     for i in range(self._n_critics):
    #         q_stds.append(self._targ_q_func.compute_q_std(
    #             batch.next_observations,
    #             self._policy.best_action(batch.next_observations),
    #         ).item())
    #
    #     q_std = torch.tensor(
    #         np.mean(q_stds), dtype=torch.float32, device=batch.device
    #     )
    #
    #     temperature = 10
    #     ratio = torch.sigmoid(-q_std * temperature) + 0.5
    #     loss = ratio.item() * self.compute_critic_loss(batch, q_tpn)
    #     loss.backward()
    #     self._critic_optim.step()
    #
    #     return loss.cpu().detach().numpy()

    # def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
    #     actions = []
    #     q_means = []
    #     q_stds = []
    #     for i in range(self._n_critics):
    #         action = self._policy.best_action(x)
    #         actions.append(action)
    #         q_means.append(self._q_func(x.unsqueeze(0), action.unsqueeze(0), reduction="mean").item())
    #         q_stds.append(self._q_func(x.unsqueeze(0), action.unsqueeze(0), reduction="std").item())
    #     lamda = 10
    #     index = torch.argmax(torch.from_numpy(np.array(q_means) + lamda * np.array(q_stds))).item()
    #
    #     return actions[index].cpu().detach().numpy()
