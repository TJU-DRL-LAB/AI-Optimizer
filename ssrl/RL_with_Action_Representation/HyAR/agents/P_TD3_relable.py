import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_1 = nn.Linear(256, discrete_action_dim)
        self.l3_2 = nn.Linear(256, parameter_action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        discrete_action = self.max_action * torch.tanh(self.l3_1(a))
        parameter_action = self.max_action * torch.tanh(self.l3_2(a))
        return discrete_action, parameter_action


class Critic(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            discrete_action_dim,
            parameter_action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.discrete_action_dim = discrete_action_dim
        self.parameter_action_dim = parameter_action_dim

        self.action_max = torch.from_numpy(np.ones((self.discrete_action_dim,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()

        self.action_parameter_max = torch.from_numpy(np.ones((self.parameter_action_dim,))).float().to(device)
        self.action_parameter_min = -self.action_parameter_max.detach()
        # print(" self.action_parameter_max_numpy", self.action_parameter_max)
        self.action_parameter_range = (self.action_parameter_max - self.action_parameter_min)

        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # 默认3e-4

        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        all_discrete_action, all_parameter_action = self.actor(state)
        return all_discrete_action.cpu().data.numpy().flatten(), all_parameter_action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, action_rep, c_rate, recon_s_rate, batch_size=256):
        recon_s_rate = recon_s_rate * 5.0
        self.total_it += 1
        # Sample replay buffer
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        # print("discrete_emb----------",discrete_emb)
        with torch.no_grad():

            discrete_emb_ = action_rep.get_embedding(discrete_action.reshape(1,
                                                                             -1).squeeze().long()).to(device)
            # discrete relable need noise
            noise_discrete = (
                    torch.randn_like(discrete_emb_) * 0.1
            ).clamp(-self.noise_clip, self.noise_clip)
            discrete_emb_table = discrete_emb_.clamp(-self.max_action, self.max_action)
            discrete_emb_table_noise = (discrete_emb_ + noise_discrete).clamp(-self.max_action, self.max_action)

            discrete_action_old = action_rep.select_discrete_action(discrete_emb).reshape(-1, 1)
            d_new = discrete_action.cpu().numpy()
            d_old = discrete_action_old
            d_bing = (d_new == d_old) * 1
            # discrete_relable_rate
            discrete_relable_rate = sum(d_bing.reshape(1, -1)[0]) / batch_size
            d_bing = torch.FloatTensor(d_bing).float().to(device)
            discrete_emb_ = d_bing * discrete_emb + (1.0 - d_bing) * discrete_emb_table_noise
            # print("discrete_emb_final",discrete_emb_)

            predict_delta_state = action_rep.select_delta_state(state, parameter_emb, discrete_emb_table)

            # print("predict_delta_state",predict_delta_state)
            # print("state_next_state",state_next_state.cpu().numpy())
            delta_state = (np.square(predict_delta_state - state_next_state.cpu().numpy())).mean(axis=1).reshape(-1, 1)
            # delta_state=predict_delta_state-state_next_state.cpu().numpy()
            # delta_state=np.mean(delta_state, axis=1).reshape(-1, 1)
            s_bing = (abs(delta_state) < recon_s_rate) * 1
            parameter_relable_rate = sum(s_bing.reshape(1, -1)[0]) / batch_size
            s_bing = torch.FloatTensor(s_bing).float().to(device)
            # print("s_bing",s_bing)

            recon_c, recon_s, mean, std = action_rep.vae(state, discrete_emb_table, parameter_action)
            parameter_emb_ = mean + std * torch.randn_like(std)
            for i in range(len(parameter_emb_[0])):
                parameter_emb_[:, i:i + 1] = self.true_parameter_emb(parameter_emb_[:, i:i + 1], c_rate, i)
            # print("parameter_emb",parameter_emb)
            # print("parameter_emb_",parameter_emb_)

            parameter_emb_ = s_bing * parameter_emb + (1 - s_bing) * parameter_emb_
            # print("parameter_emb_final",parameter_emb_)

            discrete_emb_ = discrete_emb_.clamp(-self.max_action, self.max_action)
            parameter_emb_ = parameter_emb_.clamp(-self.max_action, self.max_action)

            discrete_emb = discrete_emb_
            parameter_emb = parameter_emb_

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise_discrete = (
                    torch.randn_like(discrete_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            noise_parameter = (
                    torch.randn_like(parameter_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_discrete_action, next_parameter_action = self.actor_target(next_state)
            next_discrete_action = (next_discrete_action + noise_discrete).clamp(-self.max_action, self.max_action)
            next_parameter_action = (next_parameter_action + noise_parameter).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_action, next_parameter_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, discrete_emb, parameter_emb)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            inverting_gradients = True
            # inverting_gradients = False
            # Compute actor loss
            if inverting_gradients:
                with torch.no_grad():
                    next_discrete_action, next_parameter_action = self.actor(state)
                    action_params = torch.cat((next_discrete_action, next_parameter_action), dim=1)
                action_params.requires_grad = True
                actor_loss = self.critic.Q1(state, action_params[:, :self.discrete_action_dim],
                                            action_params[:, self.discrete_action_dim:]).mean()
            else:
                next_discrete_action, next_parameter_action = self.actor(state)
                actor_loss = -self.critic.Q1(state, next_discrete_action, next_parameter_action).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if inverting_gradients:
                from copy import deepcopy
                delta_a = deepcopy(action_params.grad.data)
                # 2 - apply inverting gradients and combine with gradients from actor
                actions, action_params = self.actor(Variable(state))
                action_params = torch.cat((actions, action_params), dim=1)
                delta_a[:, self.discrete_action_dim:] = self._invert_gradients(
                    delta_a[:, self.discrete_action_dim:].cpu(),
                    action_params[:, self.discrete_action_dim:].cpu(),
                    grad_type="action_parameters", inplace=True)
                delta_a[:, :self.discrete_action_dim] = self._invert_gradients(
                    delta_a[:, :self.discrete_action_dim].cpu(),
                    action_params[:, :self.discrete_action_dim].cpu(),
                    grad_type="actions", inplace=True)
                out = -torch.mul(delta_a, action_params)
                self.actor.zero_grad()
                out.backward(torch.ones(out.shape).to(device))
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)

            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return discrete_relable_rate, parameter_relable_rate


    # def save(self, filename, directory):
    #     torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
    #     # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))
    #
    # def load(self, filename, directory):
    #     self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
    #     # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)


    def save(self, filename,directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename,directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):

        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                index = grad[n] > 0
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]

        return grad

    def count_boundary(self, c_rate):
        median = (c_rate[0] - c_rate[1]) / 2
        offset = c_rate[0] - 1 * median
        return median, offset

    def true_parameter_emb(self, parameter_action, c_rate, i):
        # parameter_action_ = parameter_action.clone()
        median, offset = self.count_boundary(c_rate[i])
        # parameter_action_[i] = parameter_action_[i] * median + offset
        parameter_action = (parameter_action - offset) / median
        return parameter_action
