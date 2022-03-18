import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim,all_parameter_action_dim ,discrete_emb_dim, parameter_emb_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, discrete_action, parameter_action, all_parameter_action,discrete_emb, parameter_emb, next_state,state_next_state, reward, done):
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.discrete_emb[ind]).to(self.device),
            torch.FloatTensor(self.parameter_emb[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.state_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
