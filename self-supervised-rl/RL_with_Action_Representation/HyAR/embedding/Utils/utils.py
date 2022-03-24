from __future__ import print_function
import numpy as np
import torch
from torch import tensor, float32
from torch.autograd import Variable
import torch.nn as nn
import shutil
import random
from collections import deque
import itertools
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync
import importlib
from time import time
import sys
from torch.utils.data import Dataset

np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())


def importanceRatio(num, denom, mix=0, power=1):
    if mix:
        denom = (1-mix)*denom + mix*num
    ratio = np.pow(num/denom, power)
    return ratio


def save_plots(rewards, config):
    np.save(config.paths['results'] + "rewards", rewards)
    if config.debug:
        if 'Grid' in config.env_name or 'room' in config.env_name:
            # Save the heatmap
            plt.figure()
            plt.title("Exploration Heatmap")
            plt.xlabel("100x position in x coordinate")
            plt.ylabel("100x position in y coordinate")
            plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
            plt.savefig(config.paths['results'] + 'heatmap.png')
            np.save(config.paths['results'] + "heatmap", config.env.heatmap)
            config.env.heatmap.fill(0)  # reset the heat map
            plt.close()

        plt.figure()
        plt.ylabel("Total return")
        plt.xlabel("Episode")
        plt.title("Performance")
        plt.plot(rewards)
        plt.savefig(config.paths['results'] + "performance.png")
        plt.close()


def plot(rewards):
    # Plot the results
    plt.figure(1)
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Trajectories")
    plt.ylabel("Reward")
    plt.title("Baseline Reward")
    plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def custom_weight_init(self):
        # Initialize the weight values
        for m in self.modules():
            weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return


class NeuralNet_with_traces(NeuralNet):
    def __init__(self):
        super(NeuralNet_with_traces, self).__init__()
        self.e_trace = {}

    def init_traces(self, params, device):
        for name, param in params():
            # if not 'bias' in name:)
            self.e_trace[name] = torch.zeros(param.shape, dtype=float32, requires_grad=False, device=device)


    def step(self, clip_norm):
        # update eligibility traces and set the gradients accordingly
        for name, param in self.named_parameters():
            self.e_trace[name] = param.grad + self.e_trace[name] * self.config.gamma * self.config.trace_lambda
            # create a copy, otherwise resetting gradients would resets eligibility vector as well
            param.grad.data = self.e_trace[name].data.clone()

        super(NeuralNet_with_traces, self).step(clip_norm)

    def reset(self):
        for name, param in self.named_parameters():
            self.e_trace[name].zero_()


class Linear_schedule:
    def __init__(self, max_len, max=1, min=0):
        self.max_len = max_len
        self.max = max
        self.min = min
        self.temp = self.max/self.max_len

    def get(self, idx):
        return max(self.min, (self.max_len - idx) * self.temp)

class Power_schedule:
    def __init__(self, pow, max=1, min=0):
        self.pow = pow
        self.min = min
        self.temp = max

    def get(self, idx=-1):
        self.temp *= self.pow
        return max(self.min, self.temp)


def binaryEncoding(num, size):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % 2
        num = num//2
        i -= 1
    return binary

def acosh(x):
    """
    Elementwise  arc-cosh

    :param x: any shape
    :return: any shape
    """
    return torch.log(x + torch.sqrt(x**2-1))


def atanh(x):
    """
    Elementwise  arc-cosh

    :param x: any shape
    :return: any shape
    """
    return torch.log((1 + x)/(1 - x))


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def squash(x, eps = 1e-5):
    """
    Squashes each vector to ball of radius 1 - \eps

    :param x: (batch x dimension)
    :return: (batch x dimension)
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    unit = x / norm
    scale = norm**2/(1 + norm**2) - eps
    x = scale * unit

    # norm_2 = torch.sum(x**2, dim=-1, keepdim=True)
    # unit = x / torch.sqrt(norm_2)
    # scale = norm_2 / (1.0 + norm_2)    # scale \in [0, 1 - eps]
    # x = scale * unit - eps  # DO NOT DO THIS. it will make magnitude of vector consisting of all negatives larger

    return x

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # print("x",x)
    # print("y",y)

    x_norm = (x ** 2).sum(1).view(-1, 1)   #sum(1)将一个矩阵的每一行向量相加
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print("x_norm",x_norm)
    # print("y_norm",y_norm)
    y_t = torch.transpose(y, 0, 1)  #交换一个tensor的两个维度
    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    #torch.mm 矩阵a和b矩阵相乘
    # dist[dist != dist] = 0 # replace nan values with 0
    # print("dist",dist)
    return dist

def pairwise_hyp_distances(x, y, eps=1e-5):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    '''

    x_norm_2 = (x ** 2).sum(1).view(-1, 1)  # dim: N x 1
    y_norm_2 = (y ** 2).sum(1).view(1, -1)  # 1 x M
    y_t = torch.transpose(y, 0, 1)  # d x M

    # (a - b)^2 = a^2 + b^2 - 2ab
    numerator = x_norm_2 + y_norm_2 - 2.0 * torch.mm(x, y_t) + eps  # dim: Nx1 + 1xM + NxM = NxM
    denominator = torch.mm(1 - x_norm_2, 1 - y_norm_2) + eps  # dim: Nx1 x 1xM = NxM

    dist = acosh(1 + 2*numerator/denominator)

    # dist[dist != dist] = 0 # replace nan values with 0
    return dist

def hyp_distances(x, y, eps=1e-5):
    '''
    Input: x is a Nxd matrix
           y is a Nxd matirx
    '''

    x_norm_2 = (x ** 2).sum(-1)  # dim: N x 1
    y_norm_2 = (y ** 2).sum(-1)  # N x 1

    # (a - b)^2 = a^2 + b^2 - 2ab
    numerator = x_norm_2 + y_norm_2 - 2.0 * (x * y).sum(-1) + eps  # dim: Nx1 + Nx1 - Nx1 = Nx1
    denominator = (1 - x_norm_2)*(1 - y_norm_2) + eps  # dim: Nx1 x Nx1 = Nx1

    dist = acosh(1 + 2*numerator/denominator)

    # dist[dist != dist] = 0 # replace nan values with 0
    return dist

class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)

def get_var_w(shape, scale=1):
    w = torch.Tensor(shape[0], shape[1])
    w = nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('sigmoid'))
    return Variable(w.type(dtype), requires_grad=True)


def get_var_b(shape):
    return Variable(torch.rand(shape).type(dtype) / 100, requires_grad=True)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = 0#.1/ np.sqrt((fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        # m.weight.data.normal_(0.0, 0.03)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        abs_path = search(dir, name).split('/')[1:]
        pos = abs_path.index('RL')
        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param



class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using Off-policy trajectories

    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=config.device)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False, device=config.device)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype
        self.config = config

    @property
    def size(self):
        return self.length

    def reset(self):
        self.length = 0

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def batch_sample(self, batch_size, randomize=True):
        if randomize:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for ids in [indices[i:i + batch_size] for i in range(0, self.length, batch_size)]:
            yield self._get(ids)

    def sample(self, batch_size):
        count = min(batch_size, self.length)
        return self._get(np.random.choice(self.length, count))

    def add(self, s1, a1, dist, r1, s2, done):
        pos = self.length
        if self.length < self.max_len:
            self.length = self.length + 1
        else:
            pos = np.random.randint(self.max_len)

        self.s1[pos] = torch.tensor(s1, dtype=self.stype)
        self.a1[pos] = torch.tensor(a1, dtype=self.atype)
        self.dist[pos] = torch.tensor(dist)
        self.r1[pos] = torch.tensor(r1)
        self.s2[pos] = torch.tensor(s2, dtype=self.stype)
        self.done[pos] = torch.tensor(done)





class Trajectory:
    """
    Pre-allocated memory interface for storing and using on-policy trajectories

    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=config.device)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False, device=config.device)

        self.ctr = 0
        self.max_len = max_len
        self.atype = atype
        self.stype= stype
        self.config = config

    def add(self, s1, a1, dist, r1, s2, done):
        if self.ctr == self.max_len:
            # self.ctr = 0
            raise OverflowError

        self.s1[self.ctr] = torch.tensor(s1, dtype=self.stype)
        self.a1[self.ctr] = torch.tensor(a1, dtype=self.atype)
        self.dist[self.ctr] = torch.tensor(dist)
        self.r1[self.ctr] = torch.tensor(r1)
        self.s2[self.ctr] = torch.tensor(s2, dtype=self.stype)
        self.done[self.ctr] = torch.tensor(done)

        self.ctr += 1

    def reset(self):
        self.ctr = 0

    @property
    def size(self):
        return self.ctr

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def get_current_transitions(self):
        pos = self.ctr
        return self.s1[:pos], self.a1[:pos], self.dist[:pos], self.r1[:pos], self.s2[:pos], self.done[:pos]

    def get_all(self):
        return self.s1, self.a1, self.dist, self.r1, self.s2, self.done

    def get_latest(self):
        return self._get([-1])

    def batch_sample(self, batch_size, nth_return):
        # Compute the estimated n-step gamma return
        R = nth_return
        for idx in range(self.ctr-1, -1, -1):
            R = self.r1[idx] + self.config.gamma * R
            self.r1[idx] = R

        # Genreate random sub-samples from the trajectory
        perm_indices = np.random.permutation(self.ctr)
        for ids in [perm_indices[i:i + batch_size] for i in range(0, self.ctr, batch_size)]:
            yield self._get(ids)


class DataBuffer(Dataset):
    def __init__(self):
        self.length = 0
        self.x_dataset = []
        self.y_dataset = []

    def add(self, x, y):
        self.x_dataset.append(x)
        self.y_dataset.append(y)
        self.length += 1

    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, index):
        x = self.x_dataset[index]
        y = self.y_dataset[index]

        return x, y




if __name__ == '__main__':

    # use this to plot Ornstein Uhlenbeck random motion
    ou = OrnsteinUhlenbeckActionNoise(2)
    states0, states1  = [], []
    for i in range(1000):
        sample = ou.sample()
        states0.append(sample[0])
        states1.append(sample[1])
    import matplotlib.pyplot as plt

    # plt.plot(np.random.randn(1000)/3)
    plt.plot(states0)
    plt.plot(states1)
    plt.show()

    # print(binaryEncoding(np.array([0,0,0,0,1,0,0,0,0])))

    # m = torch.Tensor([[1, 2], [3, 4], [5,6], [7,8]])
    # ids = torch.Tensor([1, 1, 0, 0]).long()
    # print(m.gather(1, ids.view(-1, 1)))

