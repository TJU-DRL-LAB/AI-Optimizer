import numpy as np
import torch
class GoalReplayBuffer(object):
  def __init__(self, state_dim, action_dim, max_size=int(1e6)):
    self.max_size = max_size
    self.ptr = 0
    self.size = 0

    self.state = np.zeros((max_size, state_dim))
    self.action = np.zeros((max_size, action_dim))
    self.next_state = np.zeros((max_size, state_dim))
    self.goal = np.zeros((max_size, state_dim))
    self.reward = np.zeros((max_size, 1))
    self.t = np.zeros((max_size, 1))
    self.not_done = np.zeros((max_size, 1))

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def add(self, state, action, next_state, goal, reward, t, done):
    self.state[self.ptr] = state
    self.action[self.ptr] = action
    self.next_state[self.ptr] = next_state
    self.goal[self.ptr] = goal
    self.reward[self.ptr] = reward
    self.t[self.ptr] = t
    self.not_done[self.ptr] = 1. - done

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)


  def sample(self, batch_size, on_policy=False, use_torch=True):
    if on_policy:
        ind = np.random.randint(self.size - 1000, self.size, size=batch_size)
    else:
        ind = np.random.randint(0, self.size, size=batch_size)

    if use_torch:
      return (
        torch.FloatTensor(self.state[ind]).to(self.device),
        torch.FloatTensor(self.action[ind]).to(self.device),
        torch.FloatTensor(self.next_state[ind]).to(self.device),
        torch.FloatTensor(self.goal[ind]).to(self.device),
        torch.FloatTensor(self.reward[ind]).to(self.device),
        torch.FloatTensor(self.t[ind]).to(self.device),
        torch.FloatTensor(self.not_done[ind]).to(self.device)
      )
    else:
      return (
          self.state[ind],
          self.action[ind],
          self.next_state[ind],
          self.goal[ind],
          self.reward[ind],
          self.t[ind],
          self.not_done[ind]
        )
