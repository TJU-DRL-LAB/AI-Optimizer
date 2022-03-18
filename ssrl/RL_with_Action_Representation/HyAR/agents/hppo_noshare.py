import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sklearn.utils import shuffle
from torch.distributions import Categorical


# Advantage Actor-Critic algorithm (A2C)
# Paper:

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + 1e-8)).pow(2) + 2 * log_std + np.log(2 * np.pi))
    likelihood = pre_sum.sum(dim=1).view(-1, 1)
    return likelihood


class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action, ):
        super(Actor, self).__init__()

        self.l1_1 = nn.Linear(state_dim, 256)
        self.l1_2 = nn.Linear(state_dim, 256)

        self.l2_1 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)

        self.l3_1 = nn.Linear(256, discrete_action_dim)
        self.l3_2 = nn.Linear(256, parameter_action_dim)

        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros([10, parameter_action_dim]).view(-1, parameter_action_dim))

    def forward(self, x):
        # 共享部分
        x_1 = F.relu(self.l1_1(x))
        x_2 = F.relu(self.l1_2(x))

        x_1 = F.relu(self.l2_1(x_1))
        x_2 = F.relu(self.l2_2(x_2))

        # 离散
        discrete_prob = F.softmax(self.l3_1(x_1),dim=1)
        # 连续
        mu = torch.tanh(self.l3_2(x_2))
        log_std = self.log_std.sum(dim=0).view(1, -1) - 0.5
        std = torch.exp(log_std)

        return discrete_prob, mu, std, log_std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class PPO(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action, device,
                 clipping_ratio=0.2):
        self.device = device
        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.clipping_ratio = clipping_ratio



    def select_action(self, state, is_test=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            discrete_prob, mu, std, log_std = self.actor(state)
        # dist = distributions.Normal(mu, std)
        # pi = dist.sample()
        # 离散 softmax 然后在分布中选择
        discrete_action = Categorical(discrete_prob)
        discrete_action = discrete_action.sample().item()
        # 连续 均值方差分布
        # dist = distributions.Normal(mu, std)
        # pi = dist.sample()

        noise = torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)
        pi = mu + noise * std

        pi = pi.clamp(-1, 1)
        parameter_action = pi * self.max_action
        # print("parameter_action",parameter_action)
        if is_test:
            return discrete_prob.cpu().data.numpy().flatten(), discrete_action, parameter_action.cpu().data.numpy().flatten()
        else:
            logp = gaussian_likelihood(pi, mu, log_std)
            return discrete_prob.cpu().data.numpy().flatten(), discrete_action, parameter_action.cpu().data.numpy().flatten(), pi.cpu().data.numpy().flatten(), logp.cpu().data.numpy().flatten()

    # def calc_logp(self, state, action):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     discrete_prob, mu, std, log_std = self.actor(state)
    #     dist = distributions.Normal(mu, std)
    #     parameter_action_log_prob = dist.log_prob(action)
    #
    #     return discrete_prob,parameter_action_log_prob

    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().data.numpy().squeeze(0)

    def train(self, replay_buffer, batch_size=32, c_epoch=10, a_epoch=10):

        obs_buf, discrete_act_buf, parameter_act_buf, adv_buf, ret_buf, discrete_logp_buf, parameter_logp_buf = replay_buffer.get()

        c_loss_list, a_loss_list, discrete_a_loss_list, parameter_a_loss_list = [], [], [], []
        for ce in range(c_epoch):
            c_loss = self.update_v(obs_buf, ret_buf, batch_size=batch_size)
            c_loss_list.append(c_loss)

        obss = torch.FloatTensor(obs_buf).to(self.device)

        discrete_act_buf = torch.FloatTensor(discrete_act_buf).to(self.device)
        parameter_act_buf = torch.FloatTensor(parameter_act_buf).to(self.device)

        advs = torch.FloatTensor(adv_buf).view(-1, 1).to(self.device)
        discrete_logp_olds = torch.FloatTensor(discrete_logp_buf).view(-1, 1).to(self.device)
        parameter_logp_olds = torch.FloatTensor(parameter_logp_buf).view(-1, 1).to(self.device)

        for ae in range(a_epoch):
            discrete_prob, mu, std, parameter_log_std = self.actor(obss)

            # 离散
            discrete_logp_t = discrete_prob.gather(1, discrete_act_buf.long())
            discrete_ratio = torch.exp(discrete_logp_t - discrete_logp_olds)
            # print("discrete_ratio",discrete_ratio)

            discrete_L1 = discrete_ratio * advs
            discrete_L2 = torch.clamp(discrete_ratio, 1 - self.clipping_ratio, 1 + self.clipping_ratio) * advs
            discrete_a_loss = -torch.min(discrete_L1, discrete_L2).mean()
            discrete_a_loss_list.append(discrete_a_loss.cpu().data.numpy())
            # print("parameter_act_buf",parameter_act_buf)
            # 连续
            parameter_logp = gaussian_likelihood(parameter_act_buf, mu, parameter_log_std)
            # print("parameter_logp",parameter_logp)
            parameter_ratio = torch.exp(parameter_logp - parameter_logp_olds)
            # print("parameter_ratio",parameter_ratio)

            parameter_L1 = parameter_ratio * advs
            parameter_L2 = torch.clamp(parameter_ratio, 1 - self.clipping_ratio, 1 + self.clipping_ratio) * advs
            parameter_a_loss = -torch.min(parameter_L1, parameter_L2).mean()
            parameter_a_loss_list.append(parameter_a_loss.cpu().data.numpy())
            # loss 离散部分和连续部分相加
            a_loss = discrete_a_loss + parameter_a_loss
            # print("1111111",a_loss)
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            a_loss_list.append(a_loss.cpu().data.numpy())

        return sum(c_loss_list) / c_epoch, sum(a_loss_list) / a_epoch, sum(discrete_a_loss_list) / a_epoch, sum(
            parameter_a_loss_list) / a_epoch

    def update_v(self, x, y, batch_size):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // batch_size, 1)
        batch_size = x.shape[0] // num_batches
        x_train, y_train = shuffle(x, y)

        losses = 0
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size
            b_x = torch.FloatTensor(x_train[start:end]).to(self.device)
            b_y = torch.FloatTensor(y_train[start:end].reshape(-1, 1)).to(self.device)

            v_loss = F.mse_loss(self.critic(b_x), b_y)
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            losses += v_loss.cpu().data.numpy()

        return losses / num_batches

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
