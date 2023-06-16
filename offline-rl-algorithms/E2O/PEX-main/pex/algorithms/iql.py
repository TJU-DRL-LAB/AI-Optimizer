import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pex.utils.util import DEFAULT_DEVICE, update_exponential_moving_average


EXP_ADV_MAX = 100.


def expectile_loss(diff, expectile):
    # weight = torch.where(diff > 0, expectile, 1 - expectile)
    weight = torch.where(diff > 0, torch.tensor(expectile, dtype=torch.float),
                         torch.tensor(1 - expectile, dtype=torch.float))
    return (weight * (diff**2)).mean()


class IQL(nn.Module):
    def __init__(self, critic, vf, policy, optimizer_ctor, max_steps,
                 tau, beta, discount=0.99, target_update_rate=0.005, use_lr_scheduler=True):
        super().__init__()
        self.critic = critic.to(DEFAULT_DEVICE)
        self.target_critic = copy.deepcopy(critic).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_ctor(self.vf.parameters())
        self.q_optimizer = optimizer_ctor(self.critic.parameters())
        self.policy_optimizer = optimizer_ctor(self.policy.parameters())
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

    def update(self, observations, actions, next_observations, rewards, terminals):

        with torch.no_grad():
            target_q = self.target_critic.min(observations, actions)
            next_v = self.vf(next_observations)

        # Update value function
        v = self.vf(observations)
        adv = target_q.detach() - v
        v_loss = expectile_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.critic(observations, actions)

        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.target_critic, self.critic, self.target_update_rate)

        self.policy_update(observations, adv, actions)

    def policy_update(self, observations, adv, actions):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions.detach())

        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_lr_scheduler:
            self.policy_lr_schedule.step()

    def select_action(self, state, evaluate=False):
        if evaluate is False:
            action_sample, _, _ = self.policy.sample(state)
            return action_sample
        else:
            _, _, action_mode = self.policy.sample(state)
            return action_mode
