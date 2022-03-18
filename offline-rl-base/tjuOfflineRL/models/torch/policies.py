import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from .encoders import Encoder, EncoderWithAction


def squash_action(
    dist: torch.distributions.Distribution, raw_action: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    return squashed_action, log_prob


class Policy(nn.Module, metaclass=ABCMeta):  # type: ignore
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_with_log_prob(x)[0]

    @abstractmethod
    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n_with_log_prob(x, n)[0]

    @abstractmethod
    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DeterministicPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return torch.tanh(self._fc(h))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DeterministicResidualPolicy(Policy):

    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, scale: float):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), encoder.action_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        return (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def best_residual_action(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "residual policy does not support best_action"
        )

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )


class SquashedNormalPolicy(Policy):

    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
    ):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def _compute_logstd(self, h: torch.Tensor) -> torch.Tensor:
        if self._use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd

    def dist(self, x: torch.Tensor) -> Normal:
        h = self._encoder(x)
        mu = self._mu(h)
        clipped_logstd = self._compute_logstd(h)
        return Normal(mu, clipped_logstd.exp())

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if deterministic:
            # to avoid errors at ONNX export because broadcast_tensors in
            # Normal distribution is not supported by ONNX
            action = self._mu(self._encoder(x))
        else:
            dist = self.dist(x)
            action = dist.rsample()

        if with_log_prob:
            return squash_action(dist, action)

        return torch.tanh(action)

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self,
        x: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action = dist.rsample((n,))

        squashed_action_T, log_prob_T = squash_action(dist, action)

        # (n, batch, action) -> (batch, n, action)
        squashed_action = squashed_action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        return squashed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x)
        action = dist.rsample((n,))
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        h = self._encoder(x)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()

        # expand shape
        # (batch_size, action_size) -> (batch_size, N, action_size)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))

        # sample noise from Gaussian distribution
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)

        return torch.tanh(expanded_mean + noise * expanded_std)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        action = self.forward(x, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)

    def get_logstd_parameter(self) -> torch.Tensor:
        assert self._use_std_parameter
        logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
        base_logstd = self._max_logstd - self._min_logstd
        return self._min_logstd + logstd * base_logstd


class CategoricalPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) -> Categorical:
        h = self._encoder(x)
        h = self._fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)

        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        return action, log_prob

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x, deterministic=True))

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        return cast(torch.Tensor, dist.logits)
