from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_probabilistic_ensemble_dynamics_model
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import ProbabilisticEnsembleDynamicsModel
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .base import TorchImplBase


class ProbabilisticEnsembleDynamicsImpl(TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _n_ensembles: int
    _variance_type: str
    _discrete_action: bool
    _use_gpu: Optional[Device]
    _dynamics: Optional[ProbabilisticEnsembleDynamicsModel]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        n_ensembles: int,
        variance_type: str,
        discrete_action: bool,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        use_gpu: Optional[Device],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._n_ensembles = n_ensembles
        self._variance_type = variance_type
        self._discrete_action = discrete_action
        self._use_gpu = use_gpu

        # initialized in build
        self._dynamics = None
        self._optim = None

    def build(self) -> None:
        self._build_dynamics()

        self.to_cpu()
        if self._use_gpu:
            self.to_gpu(self._use_gpu)

        self._build_optim()

    def _build_dynamics(self) -> None:
        self._dynamics = create_probabilistic_ensemble_dynamics_model(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            n_ensembles=self._n_ensembles,
            discrete_action=self._discrete_action,
        )

    def _build_optim(self) -> None:
        assert self._dynamics is not None
        self._optim = self._optim_factory.create(
            self._dynamics.parameters(), lr=self._learning_rate
        )

    def _predict(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._dynamics is not None
        if indices is None:
            indices = torch.randint(self._n_ensembles, size=(x.shape[0],))
        else:
            assert indices.shape == (x.shape[0],)
        return self._dynamics.predict_with_variance(
            x,
            action,
            variance_type=self._variance_type,
            indices=indices.long(),
        )

    @train_api
    @torch_api()
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._dynamics is not None
        assert self._optim is not None

        loss = self._dynamics.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
        )

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()
