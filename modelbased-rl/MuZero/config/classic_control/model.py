import torch
import torch.nn as nn

from core.model import BaseMuZeroNet


class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.hx_size = 32
        self._representation = nn.Sequential(nn.Linear(input_size, self.hx_size),
                                             nn.Tanh())
        self._dynamics_state = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                             nn.Tanh(),
                                             nn.Linear(64, self.hx_size),
                                             nn.Tanh())
        self._dynamics_reward = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                              nn.LeakyReLU(),
                                              nn.Linear(64, reward_support_size))
        self._prediction_actor = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, action_space_n))
        self._prediction_value = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, value_support_size))
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward
