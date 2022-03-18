from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...gpu import Device
from ...models.builders import create_squashed_normal_policy
from ...models.encoders import EncoderFactory
from ...models.optimizers import AdamFactory, OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import squash_action
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .sac_impl import SACImpl


class AWACImpl(SACImpl):

    _lam: float
    _n_action_samples: int

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
        lam: float,
        n_action_samples: int,
        n_critics: int,
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
            temp_learning_rate=0.0,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=AdamFactory(),
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=1e-20,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._lam = lam
        self._n_action_samples = n_action_samples

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-6.0,
            max_logstd=0.0,
            use_std_parameter=True,
        )

    @train_api
    @torch_api()
    def update_actor(
        self, batch: TorchMiniBatch
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        # get current standard deviation for policy function for debug
        mean_std = self._policy.get_logstd_parameter().exp().mean()

        return loss.cpu().detach().numpy(), mean_std.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None

        dist = self._policy.dist(batch.observations)

        # unnormalize action via inverse tanh function
        clipped_actions = batch.actions.clamp(-0.999999, 0.999999)
        unnormalized_act_t = torch.atanh(clipped_actions)

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_act_t)

        # compute exponential weight
        weights = self._compute_weights(batch.observations, batch.actions)

        return -(log_probs * weights).sum()

    def _compute_weights(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert self._policy is not None
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # compute action-value
            q_values = self._q_func(obs_t, act_t, "min")

            # sample actions
            # (batch_size * N, action_size)
            policy_actions = self._policy.sample_n(
                obs_t, self._n_action_samples
            )
            flat_actions = policy_actions.reshape(-1, self.action_size)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, 1, obs_size)
            reshaped_obs_t = obs_t.view(batch_size, 1, *obs_t.shape[1:])
            # (batch_sie, 1, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = reshaped_obs_t.expand(
                batch_size, self._n_action_samples, *obs_t.shape[1:]
            )
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = repeated_obs_t.reshape(-1, *obs_t.shape[1:])

            # compute state-value
            flat_v_values = self._q_func(flat_obs_t, flat_actions, "min")
            reshaped_v_values = flat_v_values.view(obs_t.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self._lam, dim=0).view(-1, 1)

        return weights * adv_values.numel()
