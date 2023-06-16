import torch
import torch.nn as nn
from pex.utils.util import mlp


class DoubleCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)
        return self.q1(sa), self.q2(sa)

    def min(self, state, action):
        return torch.min(*self.forward(state, action))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)