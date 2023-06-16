from abc import abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ...gpu import Device
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import (
    eval_api,
    get_state_dict,
    map_location,
    set_state_dict,
    to_cpu,
    to_cuda,
    torch_api,
)
from ..base import DynamicsImplBase


class TorchImplBase(DynamicsImplBase):

    _observation_shape: Sequence[int]
    _action_size: int
    _scaler: Optional[Scaler]
    _action_scaler: Optional[ActionScaler]
    _reward_scaler: Optional[RewardScaler]
    _device: str

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._scaler = scaler
        self._action_scaler = action_scaler
        self._reward_scaler = reward_scaler
        self._device = "cpu:0"

    @eval_api
    @torch_api(scaler_targets=["x"], action_scaler_targets=["action"])
    def predict(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            observation, reward, variance = self._predict(x, action, indices)

            if self._scaler:
                observation = self._scaler.reverse_transform(observation)

            if self._reward_scaler:
                reward = self._reward_scaler.reverse_transform(reward)

        observation = observation.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        variance = variance.cpu().detach().numpy()

        return observation, reward, variance

    @abstractmethod
    def _predict(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def to_gpu(self, device: Device = Device()) -> None:
        self._device = f"cuda:{device.get_id()}"
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def scaler(self) -> Optional[Scaler]:
        return self._scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        return self._action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        return self._reward_scaler
