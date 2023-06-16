import math
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


def _make_taus(
    h: torch.Tensor, n_quantiles: int, training: bool
) -> torch.Tensor:
    if training:
        taus = torch.rand(h.shape[0], n_quantiles, device=h.device)
    else:
        taus = torch.linspace(
            start=0,
            end=1,
            steps=n_quantiles,
            device=h.device,
            dtype=torch.float32,
        )
        taus = taus.view(1, -1).repeat(h.shape[0], 1)
    return taus


def compute_iqn_feature(
    h: torch.Tensor,
    taus: torch.Tensor,
    embed: nn.Linear,
    embed_size: int,
) -> torch.Tensor:
    # compute embedding
    steps = torch.arange(embed_size, device=h.device).float() + 1
    # (batch, quantile, embedding)
    expanded_taus = taus.view(h.shape[0], -1, 1)
    prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
    # (batch, quantile, embedding) -> (batch, quantile, feature)
    phi = torch.relu(embed(prior))
    # (batch, 1, feature) -> (batch,  quantile, feature)
    return h.view(h.shape[0], 1, -1) * phi


class DiscreteIQNQFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        return _make_taus(h, n_quantiles, self.training)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        # (batch, quantile, feature) -> (batch, action, quantile)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
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
        taus = self._make_taus(h)
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
        taus = self._make_taus(h)
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


class ContinuousIQNQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        return _make_taus(h, n_quantiles, self.training)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        # (batch, quantile, feature) -> (batch, quantile)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
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
        taus = self._make_taus(h)
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
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder
