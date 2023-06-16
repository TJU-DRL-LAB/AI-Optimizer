import csv
from datetime import datetime
import json
import os
import pickle
from pathlib import Path
import random
import string
import sys

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_default_device():
    """Set the default device.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def to_torch_device(x_np):
    return torch.FloatTensor(x_np)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices].cuda() for k, v in dataset.items()}


def get_batch_from_buffer(memory, batch_size):
    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

    state_batch = torch.FloatTensor(state_batch).to(DEFAULT_DEVICE)
    next_state_batch = torch.FloatTensor(next_state_batch).to(DEFAULT_DEVICE)
    action_batch = torch.FloatTensor(action_batch).to(DEFAULT_DEVICE)
    reward_batch = torch.FloatTensor(reward_batch).to(DEFAULT_DEVICE)
    mask_batch = torch.FloatTensor(mask_batch).to(DEFAULT_DEVICE)
    return state_batch, action_batch, next_state_batch, reward_batch, mask_batch


def get_batch_from_dataset_and_buffer(dataset, buffer, batch_size, double_buffer):
    if double_buffer:
        half_batch_size = int(batch_size / 2)
        state_batch, action_batch, next_state_batch, reward_batch, terminals = get_batch_from_buffer(buffer, half_batch_size)

        res = sample_batch(dataset, batch_size - half_batch_size)

        state_batch0 = res['observations'].to(DEFAULT_DEVICE)
        action_batch0 = res['actions'].to(DEFAULT_DEVICE)
        reward_batch0 = res['rewards'].to(DEFAULT_DEVICE)
        next_state_batch0 = res['next_observations'].to(DEFAULT_DEVICE)
        terminals0 = res['terminals'].to(DEFAULT_DEVICE)

        state_batch = torch.cat([state_batch0, state_batch], dim=0)
        action_batch = torch.cat([action_batch0, action_batch], dim=0)
        next_state_batch = torch.cat([next_state_batch0, next_state_batch], dim=0)
        reward_batch = torch.cat([reward_batch0, reward_batch], dim=0)
        terminals = torch.cat([terminals0, terminals], dim=0)
    else:
        state_batch, action_batch, next_state_batch, reward_batch, terminals = get_batch_from_buffer(buffer, batch_size)

    return state_batch, action_batch, next_state_batch, reward_batch, terminals


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'


def get_mode(dist):
    """Get the (transformed) mode of the distribution.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1134
    """
    if isinstance(dist, td.categorical.Categorical):
        mode = torch.argmax(dist.logits, -1)
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.Independent):
        mode = get_mode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_mode(dist.base_dist)
        mode = base_mode
        for transform in dist.transforms:
            mode = transform(mode)
    return mode


def epsilon_greedy_sample(dist, eps=0.1):
    """Generate greedy sample that maximizes the probability.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1106
    """

    def greedy_fn(dist):
        greedy_action = get_mode(dist)
        if eps == 0.0:
            return greedy_action
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return dist.sample()
    else:
        return greedy_fn(dist)


def extract_sub_dict(prefix, dict):

    def _remove_prefix(s, prefix):
        if s.startswith(prefix):
            return s[len(prefix):]
        else:
            return s

    sub_dict = {
            _remove_prefix(k, prefix + '.'): v
            for k, v in dict.items() if k.startswith(prefix)
        }

    return sub_dict


def get_env_and_dataset(env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        # min_ret, max_ret = return_range(dataset, max_episode_steps)
        # reward_transformer = lambda x: x * max_episode_steps / (max_ret - min_ret)
        reward_transformer = lambda x: x
    elif 'antmaze' in env_name:
        reward_transformer = lambda x: x - 1

    dataset['rewards'] = reward_transformer(dataset['rewards'])

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, reward_transformer


def eval_policy(env, env_name, alg, max_episode_steps, n_eval_episodes, seed):
    eval_returns = np.array([evaluate_policy(env, alg, max_episode_steps) \
                                for _ in range(n_eval_episodes)])
    normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
    with open(f'./results/IQL_online_IQL_' + env_name + '_seed' + str(seed) + '/evaluation.csv', mode='a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([eval_returns.mean(), eval_returns.std(), normalized_returns.mean(), normalized_returns.std()])
    print({
        'return mean': round(eval_returns.mean(), 1),
        'return std': round(eval_returns.std(), 1),
        'normalized return mean': round(normalized_returns.mean(), 1),
        'normalized return std': round(normalized_returns.std(), 1),
    })


def evaluate_policy(env, agent, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    # for _ in range(max_episode_steps):
    done = False
    while not done:
        with torch.no_grad():
            action = agent.select_action(torchify(obs), evaluate=deterministic).detach().cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = next_obs
    return total_reward
