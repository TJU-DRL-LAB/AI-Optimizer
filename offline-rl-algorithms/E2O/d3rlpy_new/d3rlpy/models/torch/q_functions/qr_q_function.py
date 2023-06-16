from typing import Optional, cast

import torch
from torch import nn

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)


def _make_taus(h: torch.Tensor, n_quantiles: int) -> torch.Tensor:
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=h.device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


class DiscreteQRQFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(
            encoder.get_feature_size(), action_size * n_quantiles
        )

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(observations)
        taus = _make_taus(h, self._n_quantiles)
        all_quantiles = self._compute_quantiles(h, taus)
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousQRQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: EncoderWithAction
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(h))

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        h = self._encoder(observations, actions)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = _make_taus(h, self._n_quantiles)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder
