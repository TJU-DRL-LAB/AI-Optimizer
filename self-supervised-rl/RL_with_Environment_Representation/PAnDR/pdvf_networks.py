import torch.nn as nn
import torch.nn.functional as F
import torch


def _weights_init(m):
    '''
    Initialize the weights of a NN.
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def weights_init_uniform_rule(m):
    '''
    Initialize the weights of a NN.
    '''
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

class PDVF_ln(nn.Module):
    '''
    Policy-Dynamics Value Function object.
    '''
    def __init__(self, state_dim, env_emb_dim, hidden_dim,policy_emb_dim,  \
                 device=torch.device("cuda")):
        super(PDVF_ln, self).__init__()

        self.device = device
        self.num_inputs = state_dim + env_emb_dim + policy_emb_dim
        self.hidden_dim = hidden_dim
        self.policy_emb_dim = policy_emb_dim

        self.fc1 = nn.Linear(self.num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def forward(self, state, env_emb, policy_emb):
        '''
        Do a forward pass with the model.
        '''
        state_env_input = torch.cat((state, env_emb, policy_emb), dim=1)
        x = F.relu(self.fc1(state_env_input)) # change 0218 F.relu
        x = F.relu(self.fc2(x))
        val = self.fc3(x)

        return val

    def init_hidden(self, bs):
        '''
        Initialize the recurrent hidden states. 
        '''
        return torch.zeros(bs, self.hidden_dim, device=self.device, dtype=torch.float)

    def reset_parameters(self):
        '''
        Reset the model's parameters.
        '''
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)


class PDVF(nn.Module):
    '''
    Policy-Dynamics Value Function object.
    '''
    def __init__(self, state_dim, env_emb_dim, hidden_dim, policy_emb_dim=2, \
                 device=torch.device("cuda")):
        super(PDVF, self).__init__()

        self.device = device
        self.num_inputs = state_dim + env_emb_dim
        self.hidden_dim = hidden_dim
        self.policy_emb_dim = policy_emb_dim

        self.fc1 = nn.Linear(self.num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mat_head = nn.Linear(hidden_dim, int(policy_emb_dim * policy_emb_dim))
        self.diag_head = nn.Linear(hidden_dim, policy_emb_dim)
        self.lin_head = nn.Linear(hidden_dim, policy_emb_dim)

        self.reset_parameters()

    def forward(self, state, env_emb, policy_emb, states=None):
        '''
        Do a forward pass with the model.
        '''
        state_env_input = torch.cat((state, env_emb), dim=1)
        x = F.relu(self.fc1(state_env_input)) # change 0218 F.relu
        x = torch.tanh(self.fc2(x))

        self.mat = self.mat_head(x)
        self.mat = self.mat.reshape(self.mat.shape[0], self.policy_emb_dim, self.policy_emb_dim)    # L矩阵
        self.mat = torch.einsum('ijk,ikl -> ijl', self.mat, torch.transpose(self.mat,1,2))  # A矩阵
        self.lin = self.lin_head(x)

        leftmat = torch.einsum('ij, ijk -> ik', policy_emb, self.mat)   # z'Az
        quad_term = torch.einsum('ij, ij -> i', leftmat, policy_emb)
        val = quad_term

        return val

    def get_qf(self, state, env_emb):
        '''
        Get an estimate of the return given 
        a state and an environment / dynamics embedding.
        '''
        state_env_input = torch.cat((state, env_emb), dim=1)
        x = F.relu(self.fc1(state_env_input))
        x = torch.tanh(self.fc2(x))

        self.mat = self.mat_head(x)
        self.mat = self.mat.reshape(self.mat.shape[0], self.policy_emb_dim, self.policy_emb_dim)
        self.mat = torch.einsum('ijk,ikl -> ijl', self.mat, torch.transpose(self.mat,1,2))
        return self.mat

    def init_hidden(self, bs):
        '''
        Initialize the recurrent hidden states. 
        '''
        return torch.zeros(bs, self.hidden_dim, device=self.device, dtype=torch.float)

    def reset_parameters(self):
        '''
        Reset the model's parameters.
        '''
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)
