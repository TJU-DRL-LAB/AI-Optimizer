import numpy as np
import torch
from torch.autograd import Variable
from torch import tensor, float32
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from Src.Utils.utils import NeuralNet, NeuralNet_with_traces, pairwise_hyp_distances, squash, pairwise_distances
from Src.Utils.Basis import NN_Basis, NN_Basis_with_traces

def get_Policy(state_dim, config):
    if config.cont_actions:
        atype = torch.float32
        actor = Gaussian(state_dim=state_dim, config=config)
        action_size = actor.action_dim
    else:
        atype = torch.long
        action_size = 1
        if 'CL_' in config.algo_name:
            actor = Categorical_masked(state_dim=state_dim, config=config)
        else:
            actor = Categorical(state_dim=state_dim, config=config)

    return actor, atype, action_size


class Policy(NeuralNet):
    def __init__(self, state_dim, config, action_dim=None):
        super(Policy, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if action_dim is None:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = action_dim

    def init(self):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': self.config.actor_lr / 100})  # Keep learning rate of variance much lower
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Policy: ", temp)


class Policy_with_traces(NeuralNet_with_traces):
    def __init__(self, state_dim, config):
        super(Policy_with_traces, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if len(config.env.action_space.shape) > 0:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = config.env.action_space.n

    def init(self):
        self.init_traces(self.named_parameters, device=self.config.device)

        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append({'params': param, 'lr': self.config.actor_lr/100}) # Keep learning rate of variance much lower
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Policy with traces: ", temp)


class Categorical_masked(Policy_with_traces):
    def __init__(self, state_dim, config):
        super(Categorical_masked, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        self.action_array = np.arange(self.action_dim)
        self.init()

    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action(self, state, explore=0, mask=[]):
        x = self.forward(state)
        if len(mask) > 0:
            x[:, mask == False] = float('-inf')
        dist = F.softmax(x, -1)

        if np.random.rand() < explore:
            action = np.random.randint(low=0, high=self.action_dim)
        else:
            ## sample from the multinomial distribution
            # action = torch.distributions.Categorical(dist).sample()  #Makes code slow!
            probs = dist.cpu().view(-1).data.numpy()
            action = np.random.choice(self.action_array, p=probs)

            ## Take action with highest probability
            # _, action = torch.max(dist, 1) # second argument is axis
            # action = int(action.cpu().view(-1).data.numpy()[0])

        return action, dist

    def get_action_w_prob(self, state, explore=0, mask=[]):
        x = self.forward(state)
        if len(mask) > 0:
            x[:, mask == False] = float('-inf')
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_array, p=probs)

        return action, dist, dist.data[0][action]

    def get_prob(self, state, action, mask=[]):
        x = self.forward(state)
        if len(mask) > 0:
            x[:, mask == False] = float('-inf')
        dist = F.softmax(x, -1)

        return dist.gather(1, action), dist

    def get_prob_from_dist(self, dist, action):
        return dist.gather(1, action)

    def get_log_prob(self, state, action, mask=[]):
        x = self.forward(state)
        if len(mask) > 0:
            x[:, mask == False] = float('-inf')
        log_dist = F.log_softmax(x + 1e-8, -1)

        return log_dist.gather(1, action), torch.exp(log_dist)

    def get_log_prob_from_dist(self, dist, action):
        return torch.log(dist.gather(1, action) + 1e-8)

    def get_entropy_from_dist(self, dist):
        return - torch.sum(dist * torch.log(dist + 1e-8), dim=-1)


class Categorical(Policy_with_traces):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        if action_dim is not None:
            self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        # self.action_array = np.arange(self.action_dim)
        self.init()


    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        ## sample from the multinomial distribution
        # action = torch.distributions.Categorical(dist).sample()  #Makes code slow!
        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        ## Take action with highest probability
        # _, action = torch.max(dist, 1) # second argument is axis
        # action = int(action.cpu().view(-1).data.numpy()[0])

        return action, dist

    def get_action_w_prob(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        return action, dist, dist.data[0][action]

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_prob_from_dist(self, dist, action):
        return dist.gather(1, action)

    def get_log_prob(self, state, action):
        x = self.forward(state)
        log_dist = F.log_softmax(x + 1e-5, -1)
        return log_dist.gather(1, action), torch.exp(log_dist)

    def get_log_prob_from_dist(self, dist, action):
        return torch.log(dist.gather(dim=1, index=action) + 1e-5)

    def get_entropy_from_dist(self, dist):
        return - torch.sum(dist * torch.log(dist + 1e-5), dim=-1)


class Gaussian(Policy_with_traces):
    def __init__(self, state_dim, config):
        super(Gaussian, self).__init__(state_dim, config)

        # Set the ranges or the actions
        self.low, self.high = config.env.action_space.low * 1.0, config.env.action_space.high * 1.0
        self.action_low = tensor(torch.from_numpy(self.low), dtype=float32, requires_grad=False, device=config.device)
        self.action_diff = tensor(torch.from_numpy(self.high - self.low), dtype=float32, requires_grad=False, device=config.device)

        print("Action Low: {} :: Action High: {}".format(self.low, self.high))

        # Initialize network architecture and optimizer
        self.fc_mean = nn.Linear(state_dim, self.action_dim)
        # self.fc_mean.weight.data.uniform_(-0, 0)   # comment this if policy the critic deeper
        if self.config.gauss_variance > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc_var = nn.Linear(state_dim, self.action_dim)
            self.forward = self.forward_with_var
        self.init()

    def forward_wo_var(self, state):
        action_mean = F.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low  # Add lower value, not subtract
        # action_mean = self.fc_mean(state)
        var = torch.ones_like(action_mean, requires_grad=False) * self.config.gauss_variance
        return action_mean, var

    def forward_with_var(self, state):
        # tanh(x) = sigmoid(2x)*2 - 1
        # action_mean = F.sigmoid(2*self.fc_mean(state)) * self.action_diff + self.action_low  # Add lower value, not subtract
        action_mean = F.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low  # Add lower value, not subtract

        action_var = F.sigmoid(self.fc_var(state)) + 1e-2
        # action_mean = self.fc_mean(state)  # Add lower value, not subtract
        return action_mean, action_var

    def get_action(self, state, explore=0):
        mean, var = self.forward(state)
        dist = Normal(mean, var)

        if np.random.rand() < explore:
            action = np.random.uniform(low=self.low, high=self.high, size=self.action_dim)
        else:
            action = dist.sample()
            action = action.cpu().view(-1).data.numpy()
            # action = np.clip(action, self.low, self.high

        return action, dist

    def get_action_w_prob(self, state, explore=0):
        mean, var = self.forward(state)
        dist = Normal(mean, var)

        action = dist.sample()
        p = self.get_prob_from_dist(dist, action).cpu().view(-1).data.numpy()
        action = action.cpu().view(-1).data.numpy()

        return action, dist, p

    def get_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)

        return torch.exp(dist.log_prob(action)), dist

    def get_prob_from_dist(self, dist, action):
        return torch.exp(dist.log_prob(action))

    def get_log_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)

        return dist.log_prob(action), dist

    def get_log_prob_from_dist(self, dist, action):
        return dist.log_prob(action)

    def get_entropy_from_dist(self, dist):
        return dist.entropy()


class embed_Gaussian(Policy_with_traces):
    def __init__(self, state_dim, action_dim, config):
        super(embed_Gaussian, self).__init__(state_dim, config)

        # override super class variable
        self.action_dim = action_dim

        # Initialize network architecture and optimizer
        self.fc_mean = nn.Linear(state_dim, self.action_dim)
        if self.config.gauss_variance > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc_var = nn.Linear(state_dim, self.action_dim)
            self.forward = self.forward_with_var
        self.init()

    def forward_wo_var(self, state):
        mean = F.tanh(self.fc_mean(state))
        # mean = self.fc_mean(state)
        var = torch.ones_like(mean, requires_grad=False) * self.config.gauss_variance
        return mean, var

    def forward_with_var(self, state):
        mean = F.tanh(self.fc_mean(state))
        var  = F.sigmoid(self.fc_var(state)) + 1e-2
        # mean = self.fc_mean(state)
        return mean, var

    def get_action(self, state, explore=0):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        action = dist.sample()
        # action = torch.clamp(action, -1, 1) #DONT DO THIS

        return action, dist

    def get_log_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        return dist.log_prob(action), dist

    def get_log_prob_from_dist(self, dist, action):
        return dist.log_prob(action)

    def get_prob_from_dist(self, dist, action, scalar=True):
        # pdf = exp(log(pdf)) # Can be made faster by creating a new pdf function for Normal distribution
        if scalar:
            prod = torch.exp(torch.sum(dist.log_prob(action), -1, keepdim=True))
        else:
            prod = torch.exp(dist.log_prob(action))
        return prod

