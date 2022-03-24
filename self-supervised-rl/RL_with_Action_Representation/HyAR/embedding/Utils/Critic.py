import numpy as np
import torch
from torch import float32
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from Src.Utils.utils import NeuralNet, NeuralNet_with_traces


class Base_Critic(NeuralNet):
    def __init__(self, state_dim, config):
        super(Base_Critic, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        print("Critic: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = self.config.optim(self.parameters(), lr=self.config.critic_lr)


class Base_Critic_with_traces(NeuralNet_with_traces):
    def __init__(self, state_dim, config):
        super(Base_Critic_with_traces, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        self.init_traces(self.named_parameters, device=self.config.device)
        self.optim = self.config.optim(self.parameters(), lr=self.config.critic_lr)
        print("Critic with traces: ", [(name, param.shape) for name, param in self.named_parameters()])


# class Critic(Base_Critic):
class Critic_with_traces(Base_Critic_with_traces):
    def __init__(self, state_dim, config):
        super(Critic_with_traces, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, 1)
        # self.fc1.weight.data.uniform_(-0, 0)  # comment this if making the critic deeper
        self.init()

    def forward(self, x):
        x = self.fc1(x)
        return x

class Critic(Base_Critic):
    def __init__(self, state_dim, config):
        super(Critic, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, 1)
        # self.fc1.weight.data.uniform_(-0, 0)  # comment this if making the critic deeper
        self.init()

    def forward(self, x):
        x = self.fc1(x)
        return x


class Qval(Base_Critic):
    def __init__(self, state_dim, action_dim, config):
        super(Qval, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, action_dim)
        self.init()

    def forward(self, x):
        x = self.fc1(x)
        return x

