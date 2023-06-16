import torch
import torch.nn as nn
from torch.distributions import constraints
import torch.distributions as td
from pex.utils.util import mlp, get_mode


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, n_hidden=2, action_space=None, scale_distribution=False, state_dependent_std=False):
        super(GaussianPolicy, self).__init__()

        self.net = mlp([num_inputs, *([hidden_dim] * n_hidden)],
                       output_activation=nn.ReLU)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        if state_dependent_std:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        else:
            self.log_std = nn.Parameter(torch.zeros(num_actions),
                requires_grad=True)
            self.log_std_linear = lambda _: self.log_std

        self.apply(weights_init_)

        self._scale_distribution = scale_distribution
        self._state_dependent_std = state_dependent_std

        if action_space is None:
            self._action_means = torch.tensor(0.)
            self._action_magnitudes = torch.tensor(1.)
        else:
            self._action_means = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            self._action_magnitudes = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)

        if not scale_distribution:
            self._mean_transform = (
                lambda inputs: self._action_means + self._action_magnitudes
                * inputs.tanh())
        else:
            self._mean_transform = lambda x: x
            self._transforms = [
                StableTanh(),
                td.AffineTransform(
                    loc=self._action_means, scale=self._action_magnitudes, cache_size=1)
            ]

    def _normal_dist(self, means, stds):
        normal_dist = DiagMultivariateNormal(loc=means, scale=stds)
        if self._scale_distribution:
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    def forward(self, x):
        h = self.net(x)
        mean = self._mean_transform(self.mean_linear(h))
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return self._normal_dist(means=mean, stds=log_std.exp())


    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        mode = get_mode(dist)
        return action, log_prob, mode

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self.forward(obs)
            return get_mode(dist) if deterministic else dist.sample()

    def to(self, device):
        self._action_magnitudes = self._action_magnitudes.to(device)
        self._action_means = self._action_means.to(device)
        return super(GaussianPolicy, self).to(device)





class DiagMultivariateNormal(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate normal distribution with diagonal variance.

        Args:
            loc (Tensor): mean of the distribution
            scale (Tensor): standard deviation. Should have same shape as ``loc``.
        """
        # set validate_args to False here to enable the construction of Normal
        # distribution with zero scale.
        super().__init__(
            td.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1)

    @property
    def stddev(self):
        return self.base_dist.stddev





def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class StableTanh(td.Transform):
    r"""Invertible transformation (bijector) that computes :math:`Y = tanh(X)`,
    therefore :math:`Y \in (-1, 1)`.

    This can be achieved by an affine transform of the Sigmoid transformation,
    i.e., it is equivalent to applying a list of transformations sequentially:

    .. code-block:: python

        transforms = [AffineTransform(loc=0, scale=2)
                      SigmoidTransform(),
                      AffineTransform(
                            loc=-1,
                            scale=2]

    However, using the ``StableTanh`` transformation directly is more numerically
    stable.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        # We use cache by default as it is numerically unstable for inversion
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, StableTanh)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Based on https://github.com/tensorflow/agents/commit/dfb8c85a01d65832b05315928c010336df13f7b9#diff-a572e559b953f965c5c2cd1b9ded2c7b

        # 0.99999997 is the maximum value such that atanh(x) is valid for both
        # float32 and float64
        def _atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        y = torch.where(
            torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        return _atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (
            torch.log(torch.tensor(2.0, dtype=x.dtype, requires_grad=False)) -
            x - nn.functional.softplus(-2.0 * x))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StableTanh(cache_size)




