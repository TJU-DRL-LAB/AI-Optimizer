from abc import ABCMeta, abstractmethod
from typing import Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .encoders import Encoder, EncoderWithAction


class ConditionalVAE(nn.Module):  # type: ignore
    _encoder_encoder: EncoderWithAction
    _decoder_encoder: EncoderWithAction
    _beta: float
    _min_logstd: float
    _max_logstd: float

    _action_size: int
    _latent_size: int
    _mu: nn.Linear
    _logstd: nn.Linear
    _fc: nn.Linear

    def __init__(
        self,
        encoder_encoder: EncoderWithAction,
        decoder_encoder: EncoderWithAction,
        beta: float,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self._encoder_encoder = encoder_encoder
        self._decoder_encoder = decoder_encoder
        self._beta = beta
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd

        self._action_size = encoder_encoder.action_size
        self._latent_size = decoder_encoder.action_size

        # encoder
        self._mu = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        self._logstd = nn.Linear(
            encoder_encoder.get_feature_size(), self._latent_size
        )
        # decoder
        self._fc = nn.Linear(
            decoder_encoder.get_feature_size(), self._action_size
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def encode(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
        h = self._encoder_encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return torch.tanh(self._fc(h))

    def decode_without_squash(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return self._fc(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + cast(torch.Tensor, self._beta * kl_loss)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.randn((x.shape[0], self._latent_size), device=x.device)
        # to prevent extreme numbers
        return self.decode(x, latent.clamp(-0.5, 0.5))

    def sample_n(
        self, x: torch.Tensor, n: int, with_squash: bool = True
    ) -> torch.Tensor:
        flat_latent_shape = (n * x.shape[0], self._latent_size)
        flat_latent = torch.randn(flat_latent_shape, device=x.device)
        # to prevent extreme numbers
        clipped_latent = flat_latent.clamp(-0.5, 0.5)

        # (batch, obs) -> (n, batch, obs)
        repeated_x = x.expand((n, *x.shape))
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = repeated_x.reshape(-1, *x.shape[1:])

        if with_squash:
            flat_actions = self.decode(flat_x, clipped_latent)
        else:
            flat_actions = self.decode_without_squash(flat_x, clipped_latent)

        # (n * batch, action) -> (n, batch, action)
        actions = flat_actions.view(n, x.shape[0], -1)

        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n(x, n, with_squash=False)


class Imitator(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    @abstractmethod
    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass


class DiscreteImitator(Imitator):
    _encoder: Encoder
    _beta: float
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, beta: float):
        super().__init__()
        self._encoder = encoder
        self._beta = beta
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_log_probs_with_logits(x)[0]

    def compute_log_probs_with_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x)
        logits = self._fc(h)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, logits

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        log_probs, logits = self.compute_log_probs_with_logits(x)
        penalty = (logits**2).mean()
        return F.nll_loss(log_probs, action.view(-1)) + self._beta * penalty


class DeterministicRegressor(Imitator):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        h = self._fc(h)
        return torch.tanh(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(self.forward(x), action)


class ProbablisticRegressor(Imitator):
    _min_logstd: float
    _max_logstd: float
    _encoder: Encoder
    _mu: nn.Linear
    _logstd: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
    ):
        super().__init__()
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._encoder = encoder
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) -> Normal:
        h = self._encoder(x)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        mu = self._mu(h)
        return torch.tanh(mu)

    def sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x)
        actions = cast(torch.Tensor, dist.rsample((n,)))
        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        dist = self.dist(x)
        return F.mse_loss(torch.tanh(dist.rsample()), action)
