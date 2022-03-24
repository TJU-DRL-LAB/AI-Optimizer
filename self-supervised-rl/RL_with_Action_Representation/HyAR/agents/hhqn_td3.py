import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, discrete_action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        discrete_action = self.max_action * torch.tanh(self.l3(a))

        return discrete_action


class ParamActor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action):
        super(ParamActor, self).__init__()

        self.l1 = nn.Linear(state_dim + discrete_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, parameter_action_dim)

        self.max_action = max_action

    def forward(self, state, discrete_action):
        sa = torch.cat([state, discrete_action], 1)
        a = F.relu(self.l1(sa))
        a = F.relu(self.l2(a))
        parameter_action = self.max_action * torch.tanh(self.l3(a))
        return parameter_action


class Critic(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)

        return q1, q2

    def Q1(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class hhqn(object):
    def __init__(self, state_dim,
                 discrete_action_dim,
                 parameter_action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.001
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

        # 上层离散动作
        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # 下层连续动作参数
        self.paramactor = ParamActor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.paramactor_target = copy.deepcopy(self.paramactor)
        self.paramactor_optimizer = torch.optim.Adam(self.paramactor.parameters(), lr=3e-4)

        # critic
        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=3e-4)

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        all_discrete_action = self.actor(state)
        all_parameter_action = self.paramactor(state, all_discrete_action)
        return all_discrete_action.cpu().data.numpy().flatten(), all_parameter_action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, _, reward, not_done = replay_buffer.sample(
            batch_size)
        # Compute the target Q value
        next_discrete_action = self.actor_target(next_state)
        next_parameter_action = self.paramactor_target(next_state, next_discrete_action)
        target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_action, next_parameter_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q1, current_Q2 = self.critic(state, discrete_emb, parameter_emb)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        inverting_gradients = True
        # inverting_gradients = False
        # Compute actor loss
        if inverting_gradients:
            with torch.no_grad():
                next_discrete_action = self.actor(state)
                next_parameter_action = self.paramactor(next_state, next_discrete_action)

                action_params = torch.cat((next_discrete_action, next_parameter_action), dim=1)
            action_params.requires_grad = True
            actor_loss = self.critic.Q1(state, action_params[:, :self.discrete_action_dim],
                                     action_params[:, self.discrete_action_dim:]).mean()
        else:
            next_discrete_action = self.actor(state)
            next_parameter_action = self.paramactor(next_state, next_discrete_action)
            actor_loss = -self.critic.Q1(state, next_discrete_action, next_parameter_action).mean()

        # Optimize the actor
        self.paramactor_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        actor_loss.backward(retain_graph=True)

        if inverting_gradients:
            from copy import deepcopy
            delta_a = deepcopy(action_params.grad.data)
            # 2 - apply inverting gradients and combine with gradients from actor
            actions = self.actor(Variable(state))
            action_params = self.paramactor(Variable(state), Variable(actions))

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
            self.paramactor.zero_grad()

            out.backward(torch.ones(out.shape).to(device))

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            torch.nn.utils.clip_grad_norm_(self.paramactor.parameters(), 10.)

        # Optimize the paramactor
        self.paramactor_optimizer.step()
        # Optimize the actor
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
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
