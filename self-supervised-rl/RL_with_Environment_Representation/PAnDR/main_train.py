import os, random, sys
import gym
import numpy as np
import pdvf_utils
import env_utils

import embedding_networks
import myant
import myswimmer
import myspaceship

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from pandr_storage import ReplayMemoryPDVF_vl
from pdvf_networks import PDVF, PDVF_ln

from pandr_arguments import get_args

from ppo.model import Policy
from ppo.envs import make_vec_envs

import env_utils
import pandr_utils
import train_utils

import myant
import myswimmer
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import random
import time
from numbers import Number
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# FIXME 0321 target network update
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# FIXME 0305 CLUB
class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBpengyi(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        N, units = mu.size()
        mask = torch.eye(N)
        unmask = 1 - mask
        unmask = unmask.unsqueeze(-1)

        # log of conditional probability of negative sample pairs
        negative = - (((y_samples_1 - prediction_1) ** 2) * unmask).sum(dim=1) / (N - 1) / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


def _get_triplet_mask(labels, dev):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device(dev)

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    # print(j_not_equal_k.shape)
    # print(i_not_equal_k.shape)
    # print(i_not_equal_j.shape)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    # print((i_equal_k ^ 1))
    valid_labels = i_equal_j * (i_equal_k ^ True)

    mask = distinct_indices * valid_labels  # Combine the two masks

    return mask


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    # got the dot product between all embeddings
    cor_mat = torch.matmul(x, x.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    norm_mat = cor_mat.diag()

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.relu(distances)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _weights_init(m):
    '''
    Initialize the weights of a NN.
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class q_net(nn.Module):
    '''
    Policy-Dynamics Value Function object.
    '''

    def __init__(self, input_dim, hidden_dim, embedding_dim, \
                 device=torch.device("cuda")):
        super(q_net, self).__init__()

        self.device = device
        self.num_inputs = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = self.embedding_dim * 2

        self.fc1 = nn.Linear(self.num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
        self.embedding_train = True

        self.reset_parameters()

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_()).to(self.device)
        if self.embedding_train:
            return mu + eps * std
        else:
            return mu

    def reset_parameters(self):
        '''
        Reset the model's parameters.
        '''
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, env_emb, policy_emb):
        '''
        Do a forward pass with the model.
        '''
        state_env_input = torch.cat((env_emb, policy_emb), dim=1).to(self.device)

        x = F.relu(self.fc1(state_env_input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        self.mu = x[..., :self.embedding_dim]
        self.std = F.softplus(x[:, self.embedding_dim:] - 5, beta=1)
        z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]  # 
        z = torch.stack(z)

        return z


# enc_input_size和env的一致 因为这里继需要env又需要policy，所以和env这个大的一样
# z_b的学习应该不需要IB了,所以这里额外添加了默认ib=false
class policy_env_embedding(object):
    def __init__(self, args, train_envs, train_policies, load_data, encoder_dim, state_dim, action_dim, embedding_dim,
                 enc_input_size, pi_enc_input_size, device, tau=0.01, target_updata_freq=1):
        self.args = args
        print(self.args.kl_lambda, self.args.seed)
        self.data = load_data
        self.device = device
        self.embedding_train = True
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_ib = 0
        # todo:是否使用最大化互信息
        self.use_mi_max = self.args.use_mi_max
        # todo:是否使用最小化互信息
        self.use_mi_min = self.args.use_mi_min
        self.tau = tau
        self.target_update_freq = target_updata_freq
        print("self.use_mi_max,self.use_mi_min", self.use_mi_max, self.use_mi_min)

        if self.use_ib:
            self.encoder_output_dim = self.embedding_dim * 2
        else:
            self.encoder_output_dim = self.embedding_dim
        self.encoder = embedding_networks.make_encoder_oh(int(self.state_dim + self.action_dim + self.state_dim),
                                                          N=self.args.num_layers, \
                                                          d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                          dropout=self.args.dropout, d_emb=self.encoder_output_dim)
        self.encoder_target = embedding_networks.make_encoder_oh(int(self.state_dim + self.action_dim + self.state_dim),
                                                                 N=self.args.num_layers, \
                                                                 d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                                 dropout=self.args.dropout,
                                                                 d_emb=self.encoder_output_dim)
        embedding_networks.init_weights(self.encoder)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder.train()
        self.encoder.to(device)
        self.encoder_target.to(device)
        self.W = torch.rand(self.embedding_dim, self.embedding_dim).to(device)
        self.W.requires_grad = True
        self.W_optimizer = optim.Adam([self.W], lr=self.args.lr_dynamics)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        # max MI de approximate distribution
        self.MI_lambda = args.mi_lambda
        self.Club_lambda = args.club_lambda
        print("self.MI_lambda,self.Club_lambda", self.MI_lambda, self.Club_lambda)
        print("use_pre_train", self.args.use_pre_train)
        self.q_input_dim = args.dynamics_embedding_dim + args.policy_embedding_dim
        self.q_hidden_dim = args.z_hidden_dim
        self.q_embedding_dim = args.both_embedding_dim
        self.q_distribution = q_net(self.q_input_dim, self.q_hidden_dim, self.q_embedding_dim, device=self.device).to(
            self.device)

        # FIXME 0305 : min MI CLUB 参数以及初始化,初始参数均为原论文参数
        # 具体参数需要调整 x,y_dim
        self.x_dim = 8
        self.y_dim = 8
        print("self.x_dim,y_dim", self.x_dim, self.y_dim)
        self.club_batch_size = 64
        self.club_hidden_size = 15
        self.club_learning_rate = 0.005
        self.club_training_steps = 4000
        self.Club = CLUB(self.x_dim, self.y_dim, self.club_hidden_size).to(device)
        self.Club.train()
        self.Club_optimizer = torch.optim.Adam(self.Club.parameters(), self.club_learning_rate)

        # Loss and Optimizer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr_dynamics)
        self.q_distribution_optimizer = optim.Adam(self.q_distribution.parameters(), lr=self.args.lr_q_distribution)

        # load data
        self.state_action_state_batch = loaded['arr_0']
        self.mask_batch = loaded['arr_1']
        self.num_train_envs_ = self.args.num_envs
        self.num_train_envs = int(self.args.num_envs * 3 / 4)

        self.num_train_policies = int(self.args.num_envs * 3 / 4)
        self.num_eval_envs = int(self.args.num_envs * 1 / 4)
        self.num_eval_policies = int(self.args.num_envs * 1 / 4)

        self.train_policies = train_policies
        self.train_envs = train_envs

        self.sas_list = []
        self.mask_list = []
        self.min_length_buffer = []
        self.q_loss = []
        self.club_loss = []

        for i in range(self.args.num_envs * (self.args.num_envs + 1) + 1):
            self.sas_list.append([])
            self.mask_list.append([])
            self.min_length_buffer.append([])

        for pi in train_policies:
            for env in train_envs:
                for n_episode in range(len(self.state_action_state_batch[pi][env])):
                    self.sas_list[pi * self.num_train_envs_ + env].extend(
                        self.state_action_state_batch[pi][env][n_episode])
                    self.mask_list[pi * self.num_train_envs_ + env].extend(self.mask_batch[pi][env][n_episode])
                self.min_length_buffer[pi * self.num_train_envs_ + env] = min(200000, len(
                    self.sas_list[pi * self.num_train_envs_ + env]))
                print("self.min_length_buffer[pi*self.num_train_envs_ + env]",
                      self.min_length_buffer[pi * self.num_train_envs_ + env])

                self.sas_list[pi * self.num_train_envs_ + env] = np.stack(
                    self.sas_list[pi * self.num_train_envs_ + env]).squeeze(-1)
                print(self.sas_list[pi * self.num_train_envs_ + env].shape)

                self.mask_list[pi * self.num_train_envs_ + env] = np.stack(
                    self.mask_list[pi * self.num_train_envs_ + env]).squeeze(-1)

        self.num_train_count = 0
        self.kl_lambda = self.args.kl_lambda
        self.data = 20210605
        self.writer = SummaryWriter('tensorboard_log/st_train_both_embedding_max_min_space_0.005_' + str(
            self.args.num_epochs_emb_z) + "-" + str(self.args.both_embedding_dim) + "-" + str(
            self.args.num_t_both_embed) + '-' + str(self.data) + '/use-mimax-' +
                                    str(self.use_mi_max) + "-use-mimin-" + str(self.use_mi_min) + "-" + str(
            self.MI_lambda) + "-" +
                                    str(self.Club_lambda) + '-use-pretr' + str(self.args.use_pre_train) + '-MIN' + str(
            self.Club_lambda) + '-MAX' + str(self.MI_lambda) + '-' +
                                    str(self.args.seed) + '-' + str(self.data))

        if self.use_ib:
            self.mu = torch.zeros(self.args.batch_size_every_z, self.embedding_dim).to(self.device)
            self.vars = torch.ones(self.args.batch_size_every_z, self.embedding_dim).to(self.device)

    def infer_posterior(self, anchor, anchor_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        embedding = self.encoder(anchor.detach().to(self.device), anchor_mask.detach().to(self.device))

        if self.use_ib:
            self.mu = embedding[..., :self.args.policy_embedding_dim]
            self.std = F.softplus(embedding[:, self.args.policy_embedding_dim:] - 5, beta=1)
            # self.vars = embedding[..., self.args.policy_embedding_dim:]
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def infer_posterior_target(self, pos, pos_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        embedding = self.encoder_target(pos.detach().to(self.device), pos_mask.detach().to(self.device))

        if self.args.use_information_bottleneck:
            self.mu = embedding[..., :self.args.policy_embedding_dim]
            self.std = F.softplus(embedding[:, self.args.both_embedding_dim:] - 5, beta=1)
            # self.vars = embedding[..., self.args.policy_embedding_dim:]
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_()).to(self.device)

        if self.embedding_train:
            return mu + eps * std
        else:
            return mu

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.embedding_dim).to(self.device),
                                           torch.ones(self.embedding_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.mu), torch.unbind(self.vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def z_embedding_train(self, num_training_steps, train_policies, train_envs):
        print("z_embedding_train")
        break_loss = np.zeros(3)
        self.z_loss = []
        for epoch in range(num_training_steps):
            self.encoder.train()
            total_loss = 0
            anchor = []
            anchor_mask = []
            positive = []
            pos_mask = []
            for pi in train_policies:
                for env in train_envs:
                    # anchor index
                    index_an = np.random.randint(0, self.min_length_buffer[
                        pi * self.num_train_envs_ + env] - self.args.num_t_both_embed,
                                                 self.args.batch_size_every_z,
                                                 )  # 每个环境采样多少个anchor的index
                    index_pos = np.random.randint(0, self.min_length_buffer[
                        pi * self.num_train_envs_ + env] - self.args.num_t_both_embed,
                                                  self.args.batch_size_every_z, )
                    anchor.extend([torch.tensor(
                        self.sas_list[pi * self.num_train_envs_ + env][
                        index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(self.device) for i in
                                   range(self.args.batch_size_every_z)])
                    anchor_mask.extend(
                        [torch.tensor(self.mask_list[pi * self.num_train_envs_ + env][
                                      index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(
                            self.device) for
                            i in
                            range(self.args.batch_size_every_z)])

                    positive.extend([torch.tensor(
                        self.sas_list[pi * self.num_train_envs_ + env][
                        index_pos[i]:index_pos[i] + self.args.num_t_both_embed]).to(self.device)
                                     for i in
                                     range(self.args.batch_size_every_z)])
                    pos_mask.extend(
                        [torch.tensor(
                            self.mask_list[pi * self.num_train_envs_ + env][
                            index_pos[i]:index_pos[i] + self.args.num_t_both_embed]).to(
                            self.device) for
                            i in
                            range(self.args.batch_size_every_z)])

            anchor = torch.stack(anchor)
            positive = torch.stack(positive)

            anchor_mask = torch.stack(anchor_mask).reshape(anchor.shape[0], 1, anchor.shape[1])
            pos_mask = torch.stack(pos_mask).reshape(positive.shape[0], 1, positive.shape[1])
            # print(anchor.shape,anchor_mask.shape)
            embedding = self.infer_posterior(anchor, anchor_mask)
            embedding_pos = self.infer_posterior_target(positive, pos_mask)

            embedding = F.normalize(embedding, p=2, dim=1)
            embedding_pos = F.normalize(embedding_pos, p=2, dim=1)

            logits = self.compute_logits(embedding, embedding_pos.detach())
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            loss = self.cross_entropy_loss(logits, labels)
            self.z_loss.append(loss.item())
            total_loss += loss.item()
            total_counts = self.args.batch_size_every_z * len(self.train_policies)

            loss.backward()
            self.encoder_optimizer.step()
            self.W_optimizer.step()
            self.encoder_optimizer.zero_grad()
            self.W_optimizer.zero_grad()
            
            if epoch % self.args.log_interval == 0:
                avg_loss = total_loss / total_counts
                # avg_kl_loss = total_kl_loss / total_counts
                print("\n# Epoch %d: Z Train Loss = %.6f " % (epoch + 1, avg_loss))
                self.writer.add_scalar('z average train loss', avg_loss, global_step=self.num_train_count + 1)
                self.num_train_count += 1

            if epoch % self.target_update_freq == 0:
                soft_update_params(self.encoder, self.encoder_target, self.tau)

    def compute_logits(self, anchor, positive):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, positive.T)  # (z_dim,B)
        logits = torch.matmul(anchor, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def pre_train(self, env_encoder, policy_encoder, num_training_steps, train_policies, train_envs):
        print("pre_train")
        self.num_train_count = 0

        for epoch in range(num_training_steps):
            self.num_train_count += 1
            env_encoder.encoder.train()
            policy_encoder.encoder.train()
            total_club_train_loss = 0
            total_MI_MAX_loss = 0
            anchor = []
            anchor_mask = []

            # FIXME:每一个策略pi选择一个作为anchor，其余的随机挑选一个作为positive
            for pi in train_policies:
                for env in train_envs:
                    # anchor index
                    # TODO 将 self.args.num_t_both_embed 改为了10
                    index_an = np.random.randint(0, self.min_length_buffer[
                        pi * self.num_train_envs_ + env] - self.args.num_t_both_embed,
                                                 self.args.batch_size_every_z,
                                                 )  # 每个环境采样多少个anchor的index

                    anchor.extend([torch.tensor(
                        self.sas_list[pi * self.num_train_envs_ + env][
                        index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(self.device) for i in
                                   range(self.args.batch_size_every_z)])

                    anchor_mask.extend(
                        [torch.tensor(self.mask_list[pi * self.num_train_envs_ + env][
                                      index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(
                            self.device) for
                            i in
                            range(self.args.batch_size_every_z)])

            anchor = torch.stack(anchor)
            anchor_mask = torch.stack(anchor_mask).reshape(anchor.shape[0], 1, anchor.shape[1])
            embedding_anchor_env = env_encoder.infer_posterior(anchor, anchor_mask)
            embedding_anchor_pi = policy_encoder.infer_posterior(anchor[:, :, 0:int(self.state_dim + self.action_dim)],
                                                                 anchor_mask)

            embedding_anchor_env = F.normalize(embedding_anchor_env, p=2, dim=1)
            embedding_anchor_pi = F.normalize(embedding_anchor_pi, p=2, dim=1)

            total_counts = self.args.batch_size_every_z * len(self.train_policies)

            if self.use_mi_min:
                # FIXME :计算club的误差，并回传
                mi_loss = self.Club.learning_loss(x_samples=embedding_anchor_env, y_samples=embedding_anchor_pi)
                self.Club_optimizer.zero_grad()
                mi_loss.backward(retain_graph=True)
                self.Club_optimizer.step()
                total_club_train_loss += mi_loss
                self.club_loss.append(mi_loss.item())
                if epoch % self.args.log_interval == 0:
                    print("\n# Pre MI CLUB  train Epoch %d: loss = %.6f" % (
                        epoch + 1, total_club_train_loss / total_counts))
                    self.writer.add_scalar(' Pre Club MI loss', total_club_train_loss / total_counts,
                                           global_step=self.num_train_count + 1)

            if self.use_mi_max:
                z_embedding = self.infer_posterior(anchor, anchor_mask)
                z_embedding = F.normalize(z_embedding, p=2, dim=1)
                approximate_z = self.q_distribution(embedding_anchor_env, embedding_anchor_pi)
                l2_loss = nn.MSELoss()
                MI_MAX_loss = l2_loss(z_embedding, approximate_z)
                self.q_loss.append(MI_MAX_loss.item())
                total_MI_MAX_loss += MI_MAX_loss.item()
                if epoch % self.args.log_interval == 0:
                    print("\n# pre MI MAX train Epoch %d: MI_MAX_loss = %.6f" % (
                        epoch + 1, total_MI_MAX_loss / total_counts))
                    self.writer.add_scalar('pre average MI MAX loss', total_MI_MAX_loss / total_counts,
                                           global_step=self.num_train_count + 1)
                MI_MAX_loss = self.MI_lambda * MI_MAX_loss
                MI_MAX_loss.backward()
                self.q_distribution_optimizer.step()
                self.q_distribution_optimizer.zero_grad()

    def train_encoders(self, env_encoder, policy_encoder, train_policies, train_envs):
        self.num_train_count = 0
        self.mi_loss = []
        self.max_loss = []
        self.policy_loss = []
        self.env_loss = []
        for epoch in range(3000):
            self.num_train_count += 1
            env_encoder.encoder.train()
            policy_encoder.encoder.train()
            total_env_loss = 0
            total_policy_loss = 0
            total_club_train_loss = 0
            total_MI_MIN_loss = 0
            total_MI_MAX_loss = 0
            anchor = []
            anchor_1 = []

            env_pos = []
            env_pos_1 = []
            pi_pos = []
            anchor_mask = []
            anchor_mask_1 = []
            env_pos_mask = []
            env_pos_mask_1 = []
            pi_pos_mask = []
            anchor_all = []
            anchor_mask_all = []
            anchor_mask_single = []
            anchor_state = []
            anchor_target = []
            sample_policy = np.array(random.sample(self.train_policies, 15))
            sample_env = np.array(random.sample(self.train_envs, 15))
            sample_policy_pos = np.array(random.sample(self.train_policies, 15))
            sample_env_pos = np.array(random.sample(self.train_envs, 15))

            for pi in range(sample_policy.shape[0]):
                # for env in range(sample_policy.shape[0]):
                index_an = np.random.randint(0, self.min_length_buffer[
                    sample_policy[pi] * self.num_train_envs_ + sample_env[pi]] - self.args.num_t_both_embed,
                                             self.args.batch_size_every_z,
                                             )  # 每个环境采样多少个anchor的index
                index_env_pos = np.random.randint(0, self.min_length_buffer[
                    sample_policy_pos[pi] * self.num_train_envs_ + sample_env[pi]] - self.args.num_t_both_embed,
                                                  self.args.batch_size_every_z,
                                                  )
                index_policy_pos = np.random.randint(0, self.min_length_buffer[
                    sample_policy[pi] * self.num_train_envs_ + sample_env_pos[pi]] - self.args.num_t_both_embed,
                                                     self.args.batch_size_every_z,
                                                     )
                index_state = np.random.randint(0, self.args.num_t_policy_embed, 25)  # 从50个随机选10个预测

                anchor.extend([torch.tensor(
                    self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                    index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(self.device) for i in
                               range(self.args.batch_size_every_z)])
                anchor_mask.extend(
                    [torch.tensor(self.mask_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                                  index_an[i]:index_an[i] + self.args.num_t_both_embed]).to(
                        self.device) for
                        i in
                        range(self.args.batch_size_every_z)])

                anchor_1.extend([torch.tensor(
                    self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                    index_an[i]:index_an[i] + self.args.num_t_env_embed]).to(self.device) for i in
                                 range(self.args.batch_size_every_z)])
                anchor_mask_1.extend(
                    [torch.tensor(self.mask_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                                  index_an[i]:index_an[i] + self.args.num_t_env_embed]).to(
                        self.device) for
                        i in
                        range(self.args.batch_size_every_z)])

                env_pos.extend([torch.tensor(
                    self.sas_list[sample_policy_pos[pi] * self.num_train_envs_ + sample_env[pi]][
                    index_env_pos[i]:index_env_pos[i] + self.args.num_t_both_embed]).to(self.device) for i in
                                range(self.args.batch_size_every_z)])

                env_pos_mask.extend(
                    [torch.tensor(self.mask_list[sample_policy_pos[pi] * self.num_train_envs_ + sample_env[pi]][
                                  index_env_pos[i]:index_env_pos[i] + self.args.num_t_both_embed]).to(
                        self.device) for
                        i in
                        range(self.args.batch_size_every_z)])

                env_pos_1.extend([torch.tensor(
                    self.sas_list[sample_policy_pos[pi] * self.num_train_envs_ + sample_env[pi]][
                    index_env_pos[i]:index_env_pos[i] + self.args.num_t_env_embed]).to(self.device) for i in
                                  range(self.args.batch_size_every_z)])

                env_pos_mask_1.extend(
                    [torch.tensor(self.mask_list[sample_policy_pos[pi] * self.num_train_envs_ + sample_env[pi]][
                                  index_env_pos[i]:index_env_pos[i] + self.args.num_t_env_embed]).to(
                        self.device) for
                        i in
                        range(self.args.batch_size_every_z)])

                pi_pos.extend([torch.tensor(
                    self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env_pos[pi]][
                    index_policy_pos[i]:index_policy_pos[i] + self.args.num_t_both_embed]).to(self.device) for i in
                               range(self.args.batch_size_every_z)])

                pi_pos_mask.extend(
                    [torch.tensor(self.mask_list[sample_policy[pi] * self.num_train_envs_ + sample_env_pos[pi]][
                                  index_policy_pos[i]:index_policy_pos[i] + self.args.num_t_both_embed]).to(
                        self.device) for
                        i in
                        range(self.args.batch_size_every_z)])

                for j in range(10):
                    anchor_all.extend([torch.tensor(
                        self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                        index_an[0]:index_an[0] + self.args.num_t_both_embed]).to(self.device)])
                    anchor_mask_all.extend(
                        [torch.tensor(self.mask_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                                      index_an[0]:index_an[0] + self.args.num_t_both_embed]).to(
                            self.device)])
                    anchor_state.extend([torch.tensor(
                        self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                            index_an[0] + index_state[j]][0:int(self.state_dim)]).to(self.device)])

                    anchor_target.extend([torch.tensor(
                        self.sas_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                            index_an[0] + index_state[j]][
                        int(self.state_dim):int(self.state_dim + self.action_dim)]).to(self.device)])

                    anchor_mask_single.extend(
                        torch.tensor([torch.tensor(
                            self.mask_list[sample_policy[pi] * self.num_train_envs_ + sample_env[pi]][
                                index_an[0] + index_state[j]]).to(self.device)]))

            anchor = torch.stack(anchor)
            anchor_1 = torch.stack(anchor_1)  # 1代表1个transition，因为env的embedding是基于1个Transition的(pdvf里就是1个)

            pi_pos = torch.stack(pi_pos)
            env_pos = torch.stack(env_pos)
            env_pos_1 = torch.stack(env_pos_1)

            anchor_mask = torch.stack(anchor_mask).reshape(anchor.shape[0], 1, anchor.shape[1])
            anchor_mask_1 = torch.stack(anchor_mask_1).reshape(anchor_1.shape[0], 1, anchor_1.shape[1])
            env_pos_mask = torch.stack(env_pos_mask).reshape(env_pos.shape[0], 1, env_pos.shape[1])
            env_pos_mask_1 = torch.stack(env_pos_mask_1).reshape(env_pos_1.shape[0], 1, env_pos_1.shape[1])
            pi_pos_mask = torch.stack(pi_pos_mask).reshape(pi_pos.shape[0], 1, pi_pos.shape[1])

            if anchor_1.shape[1] == 1:
                anchor_1 = anchor_1.repeat(1, 2, 1)
                anchor_mask_1 = anchor_mask_1.repeat(1, 1, 2)
            if env_pos_1.shape[1] == 1:
                env_pos_1 = env_pos_1.repeat(1, 2, 1)
                env_pos_mask_1 = env_pos_mask_1.repeat(1, 1, 2)

            embedding_anchor_env = env_encoder.infer_posterior(anchor, anchor_mask)
            embedding_anchor_env_1 = env_encoder.infer_posterior(anchor_1, anchor_mask_1)

            embedding_anchor_pi = policy_encoder.infer_posterior(anchor[:, :, 0:int(self.state_dim + self.action_dim)],
                                                                 anchor_mask)

            embedding_env_pos = env_encoder.infer_posterior_target(env_pos, env_pos_mask)
            embedding_env_pos_1 = env_encoder.infer_posterior_target(env_pos_1, env_pos_mask_1)

            embedding_anchor_env = F.normalize(embedding_anchor_env, p=2, dim=1)
            embedding_anchor_env_1 = F.normalize(embedding_anchor_env_1, p=2, dim=1)
            embedding_anchor_pi = F.normalize(embedding_anchor_pi, p=2, dim=1)
            embedding_env_pos = F.normalize(embedding_env_pos, p=2, dim=1)
            embedding_env_pos_1 = F.normalize(embedding_env_pos_1, p=2, dim=1)

            logits_env = env_encoder.compute_logits(embedding_anchor_env_1, embedding_env_pos_1.detach())
            labels_env = torch.arange(logits_env.shape[0]).long().to(self.device)
            loss_env = env_encoder.cross_entropy_loss(logits_env, labels_env)
            total_env_loss += loss_env
            self.env_loss.append(loss_env.item())

            # decoder-encoder train
            anchor_all = torch.stack(anchor_all)
            anchor_mask_all = torch.stack(anchor_mask_all).reshape(anchor_all.shape[0], 1, anchor_all.shape[1])
            anchor_mask_single = torch.stack(anchor_mask_single)
            # anchor_mask_single = torch.tensor(anchor_mask_single)
            anchor_state = torch.stack(anchor_state)
            anchor_target = torch.stack(anchor_target)

            anchor_mask_single = torch.tensor(anchor_mask_single).unsqueeze(-1)

            embedding_anchor_all = policy_encoder.infer_posterior(
                anchor_all[:, :, 0:int(self.state_dim + self.action_dim)], anchor_mask_all)
            embedding_anchor_all = F.normalize(embedding_anchor_all, p=2, dim=1)
            anchor_state *= anchor_mask_single.to(self.device)  # mask掉结束状态,因为结束状态没有a s'啦

            embedding_anchor_all = embedding_anchor_all * (anchor_mask_single).to(self.device)  # z_pai　policy的embedding
            recurrent_hidden_state = torch.zeros(args.policy_batch_size,
                                                 policy_encoder.decoder.recurrent_hidden_state_size, device=self.device,
                                                 requires_grad=True).float()
            mask_dec = torch.zeros(args.policy_batch_size, 1, device=self.device, requires_grad=True).float()
            emb_state_input = torch.cat((embedding_anchor_all.detach(), anchor_state.to(self.device)), dim=1).to(
                self.device)
            action = policy_encoder.decoder(emb_state_input, recurrent_hidden_state, mask_dec)
            action *= anchor_mask_single.to(self.device)
            anchor_target *= anchor_mask_single.to(self.device)

            decoder_loss = self.criterion(action, anchor_target.to(self.device))

            total_policy_loss += decoder_loss.item()
            self.policy_loss.append(decoder_loss.item)

            total_counts = self.args.batch_size_every_z * len(self.train_policies)

            if self.use_mi_min:
                # FIXME :计算club的误差，并回传
                mi_loss = self.Club.learning_loss(x_samples=embedding_anchor_env, y_samples=embedding_anchor_pi)
                self.Club_optimizer.zero_grad()
                mi_loss.backward(retain_graph=True)
                self.Club_optimizer.step()
                total_club_train_loss += mi_loss
                if epoch % self.args.log_interval == 0:
                    print("\n# MI CLUB  train Epoch %d: loss = %.6f" % (
                        epoch + 1, total_club_train_loss / total_counts))
                    self.writer.add_scalar(' Club MI loss', total_club_train_loss / total_counts,
                                           global_step=self.num_train_count + 1)

            if epoch % self.args.log_interval == 0:
                print("\n# env embedding train Epoch %d: loss = %.6f" % (
                    epoch + 1, total_env_loss / total_counts))
                self.writer.add_scalar(' env embedding loss', total_env_loss / total_counts,
                                       global_step=self.num_train_count + 1)
            if epoch % self.args.log_interval == 0:
                self.num_train_count += 1
                print("\n# policy embedding train Epoch %d: loss = %.6f" % (
                    epoch + 1, total_policy_loss / total_counts))
                self.writer.add_scalar(' policy embedding loss', total_policy_loss / total_counts,
                                       global_step=self.num_train_count + 1)

            # MIN MI loss
            if self.use_mi_min:
                MI_MIN_loss = self.Club(x_samples=embedding_anchor_env, y_samples=embedding_anchor_pi)
                total_MI_MIN_loss += MI_MIN_loss.item()
                self.mi_loss.append(MI_MIN_loss.item())
                if epoch % self.args.log_interval == 0:
                    print("\n# embedding train Epoch %d: MI_MIN_loss = %.6f" % (
                        epoch + 1, total_MI_MIN_loss / total_counts))
                    self.writer.add_scalar('average MI MIN loss', total_MI_MIN_loss / total_counts,
                                           global_step=self.num_train_count + 1)
                MI_MIN_loss = epoch*self.Club_lambda * MI_MIN_loss
                MI_MIN_loss.backward(retain_graph=True)

            # max MI loss
            if self.use_mi_max and epoch%5==0:
                z_embedding = self.infer_posterior(anchor, anchor_mask)
                z_embedding = F.normalize(z_embedding, p=2, dim=1)
                approximate_z = self.q_distribution(embedding_anchor_env, embedding_anchor_pi)  # 这里用env_1好还是用env好？？？
                l2_loss = nn.MSELoss()
                MI_MAX_loss = l2_loss(z_embedding, approximate_z)
                total_MI_MAX_loss += MI_MAX_loss.item()
                self.max_loss.append(MI_MAX_loss.item())
                self.q_loss.append(MI_MAX_loss.item())
                if epoch % self.args.log_interval == 0:
                    print("\n# embedding train Epoch %d: MI_MAX_loss = %.6f" % (
                        epoch + 1, total_MI_MAX_loss / total_counts))
                    self.writer.add_scalar('average MI MAX loss', total_MI_MAX_loss / total_counts,
                                           global_step=self.num_train_count + 1)
                MI_MAX_loss = self.MI_lambda * MI_MAX_loss
                MI_MAX_loss.backward(retain_graph=True)
                self.q_distribution_optimizer.step()
                self.q_distribution_optimizer.zero_grad()

            decoder_loss.backward()
            loss_env.backward()
            policy_encoder.encoder_optimizer.step()
            policy_encoder.decoder_optimizer.step()
            env_encoder.encoder_optimizer.step()
            env_encoder.W_optimizer.step()
            policy_encoder.encoder_optimizer.zero_grad()
            policy_encoder.decoder_optimizer.zero_grad()
            env_encoder.encoder_optimizer.zero_grad()
            env_encoder.W_optimizer.zero_grad()

            if epoch % self.target_update_freq == 0:
                soft_update_params(env_encoder.encoder, env_encoder.encoder_target, self.tau)
                soft_update_params(policy_encoder.encoder, policy_encoder.encoder_target, self.tau)

    def save_model(self):
        # Save the models
        pdvf_utils.save_model(
            str(self.args.seed) + "-" + str(self.args.num_epochs_emb_z) + "-" + str(
                self.args.both_embedding_dim) + "-" + str(
                self.args.num_t_both_embed) + "-" + str(
                self.args.shuffle) + "-norm-cont-both-encoder.", self.encoder, self.encoder_optimizer, \
            self.num_train_count, self.args, self.args.env_name, save_dir=self.args.save_dir_both_embedding)


class policy_embedding(object):
    def __init__(self, args, train_envs, train_policies, load_data, encoder_dim, enc_input_size, embedding_dim, env,
                 device):
        self.args = args
        print(self.args.kl_lambda, self.args.seed)
        self.data = load_data
        self.device = device
        self.embedding_train = True
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        if self.args.use_information_bottleneck:
            self.encoder_output_dim = self.embedding_dim * 2
        else:
            self.encoder_output_dim = self.embedding_dim
        self.encoder = embedding_networks.make_encoder_oh(enc_input_size, N=self.args.num_layers, \
                                                          d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                          dropout=self.args.dropout, d_emb=self.encoder_output_dim)
        self.encoder_target = embedding_networks.make_encoder_oh(enc_input_size, N=self.args.num_layers, \
                                                                 d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                                 dropout=self.args.dropout,
                                                                 d_emb=self.encoder_output_dim)
        self.decoder = Policy(
            tuple([env.observation_space.shape[0] + args.policy_embedding_dim]),
            env.action_space,
            base_kwargs={'recurrent': False})
        embedding_networks.init_weights(self.encoder)
        embedding_networks.init_weights(self.decoder)
        self.criterion = nn.MSELoss(reduction='sum')
        self.encoder.train()
        self.decoder.train()
        self.encoder.to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_target.to(device)
        self.decoder.to(device)
        # Loss and Optimizer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr_dynamics)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=args.lr_policy)
        self.criterion = nn.MSELoss(reduction='sum')
        self.state_action_state_batch = loaded['arr_0']
        self.mask_batch = loaded['arr_1']
        self.num_train_envs = int(self.args.num_envs * 3 / 4)
        self.num_eval_envs = int(self.args.num_envs * 1 / 4)

        self.train_policies = train_policies
        self.train_envs = train_envs

        self.sa_list = []
        self.mask_list = []
        self.min_length_buffer = []

        for pi in range(self.args.num_envs):
            self.sa_list.append([])
            self.mask_list.append([])
            self.min_length_buffer.append([])

        for pi in train_policies:
            for env in train_envs:
                for n_episode in range(len(self.state_action_state_batch[pi][env])):
                    self.sa_list[pi].extend(self.state_action_state_batch[pi][env][n_episode])
                    self.mask_list[pi].extend(self.mask_batch[pi][env][n_episode])
            # print(len(sa_list[pi]))
            self.min_length_buffer[pi] = min(200000, len(self.sa_list[pi]))
            self.sa_list[pi] = np.stack(self.sa_list[pi]).squeeze(-1)
            # print(sa_list[pi].shape)
            self.sa_list[pi] = self.sa_list[pi][:, 0:int(enc_input_size)]
            self.mask_list[pi] = np.stack(self.mask_list[pi]).squeeze(-1)

        self.num_train_count = 0
        self.kl_lambda = self.args.kl_lambda

        # information bottleneck
        self.use_ib = self.args.use_information_bottleneck
        self.kl_lambda = self.args.kl_lambda
        if self.use_ib:
            self.mu = torch.zeros(self.args.batch_size_every_policy, self.embedding_dim).to(self.device)
            self.vars = torch.ones(self.args.batch_size_every_policy, self.embedding_dim).to(self.device)

    def infer_posterior(self, anchor, anchor_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        embedding = self.encoder(anchor.detach().to(self.device), anchor_mask.detach().to(self.device))

        if self.args.use_information_bottleneck:
            self.mu = embedding[..., :self.args.policy_embedding_dim]
            self.std = F.softplus(embedding[:, self.args.policy_embedding_dim:] - 5, beta=1)
            # self.vars = embedding[..., self.args.policy_embedding_dim:]
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def infer_posterior_target(self, pos, pos_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        embedding = self.encoder_target(pos.detach().to(self.device), pos_mask.detach().to(self.device))

        if self.args.use_information_bottleneck:
            self.mu = embedding[..., :self.args.policy_embedding_dim]
            self.std = F.softplus(embedding[:, self.args.policy_embedding_dim:] - 5, beta=1)
            # self.vars = embedding[..., self.args.policy_embedding_dim:]
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            # print(self.mu.shape,self.vars)
            # self.mu, self.std = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(std))]
            # posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
            #               zip(torch.unbind(self.mu), torch.unbind(self.vars))]
            # z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_()).to(self.device)

        if self.embedding_train:
            return mu + eps * std
        else:
            return mu

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.embedding_dim).to(self.device),
                                           torch.ones(self.embedding_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.mu), torch.unbind(self.vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def save_model(self):
        # Save the models
        pdvf_utils.save_model(
            str(self.args.seed) + "-" + str(self.args.num_epochs_emb_policy) + "-" + str(
                self.args.policy_embedding_dim) + "-" + str(
                self.args.num_t_policy_embed) + "-" + str(
                self.args.shuffle) + "-norm-cont-policy-encoder.", self.encoder, self.encoder_optimizer, \
            self.num_train_count, self.args, self.args.env_name, save_dir=self.args.save_dir_policy_embedding)


class env_embedding(object):
    def __init__(self, args, train_envs, train_policies, load_data, encoder_dim, enc_input_size, embedding_dim, device):
        self.args = args
        print(self.args.kl_lambda, self.args.seed)
        self.data = load_data
        self.device = device
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.embedding_train = True
        if self.args.use_information_bottleneck:
            self.encoder_output_dim = self.embedding_dim * 2
        else:
            self.encoder_output_dim = self.embedding_dim
        self.encoder = embedding_networks.make_encoder_oh(enc_input_size, N=self.args.num_layers, \
                                                          d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                          dropout=self.args.dropout, d_emb=self.encoder_output_dim)
        self.encoder_target = embedding_networks.make_encoder_oh(enc_input_size, N=self.args.num_layers, \
                                                                 d_model=self.encoder_dim, h=self.args.num_attn_heads, \
                                                                 dropout=self.args.dropout,
                                                                 d_emb=self.encoder_output_dim)
        embedding_networks.init_weights(self.encoder)
        self.encoder.train()
        self.encoder.to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_target.to(device)
        self.W = torch.rand(self.embedding_dim, self.embedding_dim).to(device)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.W.requires_grad = True
        self.W_optimizer = optim.Adam([self.W], lr=self.args.lr_dynamics)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr_dynamics)
        self.state_action_state_batch = loaded['arr_0']
        self.mask_batch = loaded['arr_1']
        self.num_train_envs = int(self.args.num_envs * 3 / 4)
        self.num_eval_envs = int(self.args.num_envs * 1 / 4)

        self.min_length_buffer = []
        self.sas_list = []
        self.mask_list = []

        for i in range(self.args.num_envs):
            self.sas_list.append([])
            self.mask_list.append([])
            self.min_length_buffer.append([])

        self.train_policies = train_policies
        self.train_envs = train_envs

        for env in self.train_envs:
            for pi in self.train_policies:
                for n_episode in range(len(self.state_action_state_batch[pi][env])):
                    self.sas_list[env].extend(self.state_action_state_batch[pi][env][n_episode])
                    self.mask_list[env].extend(self.mask_batch[pi][env][n_episode])
            # print(len(sa_list[pi]))
            self.min_length_buffer[env] = min(200000, len(self.sas_list[env]))
            self.sas_list[env] = np.stack(self.sas_list[env]).squeeze(-1)
            # print(sa_list[pi].shape)
            self.sas_list[env] = self.sas_list[env]
            self.mask_list[env] = np.stack(self.mask_list[env]).squeeze(-1)

        self.num_train_count = 0
        self.kl_lambda = self.args.kl_lambda

        self.use_ib = self.args.use_information_bottleneck
        self.max_MI = self.args.max_mutual_information

        if self.use_ib:
            self.mu = torch.zeros(self.args.batch_size_every_env, self.embedding_dim).to(self.device)
            self.vars = torch.ones(self.args.batch_size_every_env, self.embedding_dim).to(self.device)

    def infer_posterior(self, anchor, anchor_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        embedding = self.encoder(anchor.detach().to(self.device), anchor_mask.detach().to(self.device))

        if self.args.use_information_bottleneck:
            self.mu = embedding[..., :self.embedding_dim]
            self.std = F.softplus(embedding[:, self.embedding_dim:] - 5, beta=1)
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def infer_posterior_target(self, pos, pos_mask):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        embedding = self.encoder_target(pos.detach().to(self.device), pos_mask.detach().to(self.device))

        if self.args.use_information_bottleneck:
            self.mu = embedding[..., :self.embedding_dim]
            self.std = F.softplus(embedding[:, self.embedding_dim:] - 5, beta=1)
            z = [self.reparametrize_n(m, s, 1) for m, s in zip(torch.unbind(self.mu), torch.unbind(self.std))]
            z = torch.stack(z)
        else:
            z = embedding

        return z

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_()).to(self.device)
        if self.embedding_train:
            return mu + eps * std
        else:
            return mu

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.embedding_dim).to(self.device),
                                           torch.ones(self.embedding_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.mu), torch.unbind(self.vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def compute_logits(self, anchor, positive):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, positive.T)  # (z_dim,B)
        logits = torch.matmul(anchor, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_logits(self, anchor, positive):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, positive.T)  # (z_dim,B)
        logits = torch.matmul(anchor, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def save_model(self):
        pdvf_utils.save_model(
            str(self.args.seed) + "-" + str(self.args.num_epochs_emb_env) + "-" + str(
                self.args.dynamics_embedding_dim) + "-" + str(
                self.args.num_t_env_embed) + "-" + str(self.args.shuffle) + "-nonorm-dynamics-encoder.", self.encoder,
            self.encoder_optimizer, \
            self.num_train_count + 1, self.args, self.args.env_name, save_dir=self.args.save_dir_dynamics_embedding)


def train_pdvf_ds(all_envs, all_policies_index, load_data, args):
    '''
    Train the Policy-Dynamics Value Function of PD-VF
    which estimates the return for a family of policies
    in a family of environments with varying dynamics.

    To do this, it trains a network conditioned on an initial state,
    a (learned) policy embedding, and a (learned) dynamics embedding
    and outputs an estimate of the cumulative reward of the
    corresponding policy in the given environment.
    '''
    # args = get_args()
    print(torch.cuda.is_available())
    os.environ['OMP_NUM_THREADS'] = '1'
    norm = True
    device = torch.device("cuda")
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(args.kl_lambda, args.seed)
    print(device)
    # Create the environment
    envs = make_vec_envs(args, device)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    writer = SummaryWriter(
        'tensorboard_log/ant_' + str(args.num_epochs_pdvf_phase1) + "_" + str(args.shuffle) + "-use-ib-" + str(
            args.use_information_bottleneck) + "-use-mimax-" + str(args.use_mi_max) + "-use-mimin-" + str(
            args.use_mi_min)
        + "-" + str(args.dynamics_embedding_dim) + "-" + str(args.policy_embedding_dim) + "-" +
        str(args.both_embedding_dim) + '-0322-st/' + str(args.kl_lambda) + '-' + str(args.seed))

    # Instantiate the cl
    args.use_l2_loss = True
    value_net = PDVF_ln(envs.observation_space.shape[0], args.dynamics_embedding_dim, args.hidden_dim_pdvf,
                        args.policy_embedding_dim, device=device).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=args.lr_pdvf, eps=args.eps)
    optimizer2 = optim.Adam(value_net.parameters(), lr=args.lr_pdvf, eps=args.eps)
    network = {'net': value_net, 'optimizer': optimizer, 'optimizer2': optimizer2}
    value_net.train()

    # load data
    # TODO:更改环境和策略的数量
    args.num_envs = 20
    num_train_envs = int(args.num_envs * 3 / 4)
    train_policies = all_policies_index[0:num_train_envs]
    train_envs = all_envs[0:num_train_envs]
    eval_envs = all_envs[num_train_envs:args.num_envs]
    eval_policies = all_policies_index[num_train_envs: args.num_envs]
    print("train_policies, eval_policies", train_policies, eval_policies,args.max_num_steps)

    NUM_TRAIN_SAMPLES = args.batch_size_every_train * len(train_policies) * len(train_envs)
    NUM_EVAL_EPS = args.num_eval_eps

    NUM_TEST_SAMPLES = args.batch_size_every_test * (
            len(eval_envs) * len(train_policies) + len(train_envs) * len(eval_policies) + len(eval_envs) * len(
        eval_policies))

    env_enc_input_size = 2 * envs.observation_space.shape[0] + args.policy_embedding_dim
    sizes = pandr_utils.DotDict({'state_dim': envs.observation_space.shape[0], \
                                    'action_dim': envs.action_space.shape[0], 'env_enc_input_size': env_enc_input_size, \
                                    'env_max_seq_length': args.max_num_steps * env_enc_input_size})

    loaded = load_data
    state_action_state_batch = loaded["arr_0"]
    mask_batch = loaded["arr_1"]
    reward_list = loaded["arr_2"]
    init_obs_list = loaded["arr_3"]

    # init encoder
    enc_input_size = 2 * envs.observation_space.shape[0] + envs.action_space.shape[0]
    pi_enc_input_size = envs.observation_space.shape[0] + envs.action_space.shape[0]
    print("enc_input_size,pi_enc_input_size", enc_input_size, pi_enc_input_size)
    # env & policy
    encoder_dim = args.num_attn_heads * args.both_attn_head_dim
    embedding_dim = args.both_embedding_dim
    policy_env_encoder = policy_env_embedding(args, train_envs, train_policies, load_data, encoder_dim,
                                              envs.observation_space.shape[0], envs.action_space.shape[0],
                                              embedding_dim, enc_input_size, pi_enc_input_size, device)

    # env
    encoder_dim = args.num_attn_heads * args.dynamics_attn_head_dim
    embedding_dim = args.dynamics_embedding_dim
    env_encoder = env_embedding(args, train_envs, train_policies, load_data, encoder_dim, enc_input_size, embedding_dim,
                                device)
    # policy
    encoder_dim = args.num_attn_heads * args.policy_attn_head_dim
    embedding_dim = args.policy_embedding_dim
    policy_encoder = policy_embedding(args, train_envs, train_policies, load_data, encoder_dim, pi_enc_input_size,
                                      embedding_dim, envs, device)

    num_eval_envs = int(args.num_envs - num_train_envs)
    print(num_train_envs, num_eval_envs)
    ####################    TRAIN PHASE 1      ########################
    # Collect test Data in unseen evn&policy
    print("\nFirst Training Stage")
    print("\n init train embedding")
    if args.max_mutual_information:
        # TODO：args.num_epochs_emb_z=10 方便测试
        # pre train z
        policy_env_encoder.z_embedding_train(num_training_steps=args.num_epochs_emb_z, train_policies=train_policies,
                                             train_envs=train_envs)
        # pre train club
        if args.use_pre_train:
            policy_env_encoder.pre_train(env_encoder, policy_encoder, num_training_steps=args.num_epochs_emb_z,
                                         train_policies=train_policies, train_envs=train_envs)
        # policy_env_encoder.q_distribution.embedding_train = False
        policy_env_encoder.embedding_train = False
        # train encoder with max MI
        policy_env_encoder.train_encoders(env_encoder, policy_encoder, train_policies=train_policies,
                                          train_envs=train_envs)
    else:
        env_encoder.env_embedding_train(num_training_steps=args.num_epochs_emb_env)
        policy_encoder.policy_embedding_train(num_training_steps=args.num_epochs_emb_policy)
        env_encoder.embedding_train = False
        policy_encoder.embedding_train = False

    policy_encoder.tsne_data_sample()
    env_encoder.tsne_data_sample()
    # normalize
    #    max_episode_reward = 1
    #    for i in range(args.batch_size_every_test):
    #        for env in all_envs:
    #            for pi in all_policies_index:
    #                max_episode_reward = max(max_episode_reward, reward_list[pi][env][i])
    #    if max_episode_reward <= 0:
    #        print("max reward wrong", max_episode_reward)

    memory = ReplayMemoryPDVF_vl(NUM_TRAIN_SAMPLES)
    test_memory = ReplayMemoryPDVF_vl(NUM_TEST_SAMPLES)
    for i in range(args.batch_size_every_test):
        for env in eval_envs:
            for pi in train_policies:
                # print(env,pi)
                index = i
                episode_reward_tensor = torch.tensor([reward_list[pi][env][index]],
                                                     device=device, dtype=torch.float)
                policy_data = np.array(state_action_state_batch[pi][env][index])
                # print(policy_data.shape)
                env_data = np.array(state_action_state_batch[pi][env][index][0:args.num_t_env_embed])
                policy_mask = np.array(mask_batch[pi][env][index])
                env_mask = np.array(mask_batch[pi][env][index][0:args.num_t_env_embed])
                policy_data = policy_data[:, 0:int(pi_enc_input_size), :]
                test_memory.push(torch.FloatTensor(init_obs_list[pi][env][index]).unsqueeze(0), policy_data,
                                 policy_mask,
                                 env_data, env_mask, episode_reward_tensor)

    for i in range(args.batch_size_every_test):
        for env in train_envs:
            for pi in eval_policies:
                index = i
                episode_reward_tensor = torch.tensor([reward_list[pi][env][index]],
                                                     device=device, dtype=torch.float)

                policy_data = np.array(state_action_state_batch[pi][env][index])
                # print(policy_data.shape)
                env_data = np.array(state_action_state_batch[pi][env][index][0:args.num_t_env_embed])
                policy_mask = np.array(mask_batch[pi][env][index])
                env_mask = np.array(mask_batch[pi][env][index][0:args.num_t_env_embed])
                policy_data = policy_data[:, 0:int(pi_enc_input_size), :]
                test_memory.push(torch.FloatTensor(init_obs_list[pi][env][index]).unsqueeze(0), policy_data,
                                 policy_mask,
                                 env_data, env_mask, episode_reward_tensor)

    for i in range(args.batch_size_every_test):
        for env in eval_envs:
            for pi in eval_policies:
                episode_reward_tensor = torch.tensor([reward_list[pi][env][index]],
                                                     device=device, dtype=torch.float)
                policy_data = np.array(state_action_state_batch[pi][env][index])
                # print(policy_data.shape)
                env_data = np.array(state_action_state_batch[pi][env][index][0:args.num_t_env_embed])
                policy_mask = np.array(mask_batch[pi][env][index])
                env_mask = np.array(mask_batch[pi][env][index][0:args.num_t_env_embed])
                policy_data = policy_data[:, 0:int(pi_enc_input_size), :]
                test_memory.push(torch.FloatTensor(init_obs_list[pi][env][index]).unsqueeze(0), policy_data,
                                 policy_mask,
                                 env_data, env_mask, episode_reward_tensor)
    print("test memory done")

    # Collect eval Data in training env&policy
    for i in range(args.batch_size_every_train):
        for env in train_envs:
            for pi in train_policies:
                index = i
                episode_reward_tensor = torch.tensor([reward_list[pi][env][index]],
                                                     device=device, dtype=torch.float)
                # print(state_action_state_batch[pi][env][index])
                policy_data = np.array(state_action_state_batch[pi][env][index])
                # print(policy_data.shape)
                env_data = np.array(state_action_state_batch[pi][env][index][0:args.num_t_env_embed])
                policy_mask = np.array(mask_batch[pi][env][index])
                env_mask = np.array(mask_batch[pi][env][index][0:args.num_t_env_embed])
                policy_data = policy_data[:, 0:int(pi_enc_input_size), :]
                memory.push(torch.FloatTensor(init_obs_list[pi][env][index]).unsqueeze(0), policy_data, policy_mask,
                            env_data, env_mask, episode_reward_tensor)

    print("training memory done")

    ### Train - Stage 1 ###
    total_train_loss = 0
    total_eval_loss = 0
    BEST_EVAL_LOSS = sys.maxsize
    total_test_loss = 0
    DECODER_BEST_EVAL_LOSS = sys.maxsize
    train_loss_list =[]
    test_loss_list = []
    print("\n train embedding and value function")
    for i in range(args.num_epochs_pdvf_phase1):
        train_loss = train_utils.optimize_model_cl(args, network,
                                                   memory, env_encoder, policy_encoder,
                                                   num_opt_steps=args.num_opt_steps,
                                                   vl_train=args.value_loss_train_embed)
        test_loss = train_utils.optimize_model_cl(args, network,
                                              test_memory, env_encoder, policy_encoder,
                                              num_opt_steps=args.num_opt_steps, eval=True,
                                              vl_train=args.value_loss_train_embed)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    if train_loss:
        total_train_loss += train_loss
        # print(test_loss)
    if test_loss:
        total_test_loss += test_loss
        if test_loss < BEST_EVAL_LOSS:
            BEST_EVAL_LOSS = test_loss
            pandr_utils.save_model("pdvf-stage0.", value_net, optimizer, \
                                   i, args, args.env_name, save_dir=args.save_dir_pdvf)
    if i % args.log_interval == 0:
        print("\n### PD-VF: Episode {}: Train Loss {:.6f}Test Loss {:.6f}".format( \
            i, total_train_loss / args.log_interval, total_test_loss / args.log_interval))
        writer.add_scalar('total_train_loss', total_train_loss, global_step=i + 1)
        # writer.add_scalar('total_eval_loss', total_eval_loss, global_step=i + 1)
        writer.add_scalar('total_test_loss', total_test_loss, global_step=i + 1)


    print("-------------------eval-------------------")
    # Eval on Train Envs
    value_net.eval()
    policy_encoder.decoder.eval()
    train_rewards = {}
    accurancy = np.zeros((args.num_envs, args.num_envs))
    accurancy_mse = np.zeros((args.num_envs, args.num_envs))
    for env in all_envs:
        for pi in all_policies_index:
            acc = 0
            acc_mse = 0
            for i in range(10):
                min_length_buffer = min(20000, len(state_action_state_batch[pi][env]))
                print(env, pi)
                index = np.random.randint(0, min_length_buffer, 1)
                index = index[0]
                episode_reward_tensor = torch.tensor(reward_list[pi][env][index],
                                                     device=device, dtype=torch.float)

                p = np.array(state_action_state_batch[pi][env][index])
                # print(policy_data.shape)
                e = np.array(state_action_state_batch[pi][env][index][0:args.num_t_env_embed])
                pm = np.array(mask_batch[pi][env][index])
                em = np.array(mask_batch[pi][env][index][0:args.num_t_env_embed])
                p = p[:, 0:int(pi_enc_input_size), :]
                p = torch.FloatTensor(p).to(device)
                e = torch.FloatTensor(e).to(device)
                pm = torch.FloatTensor(pm).to(device)
                em = torch.FloatTensor(em).to(device)
                policy_data = p.squeeze(-1).unsqueeze(0)
                env_data = e.squeeze(-1).unsqueeze(0)
                policy_mask = pm.squeeze(-1).unsqueeze(0).unsqueeze(0)
                env_mask = em.unsqueeze(0)
                if env_data.shape[1] == 1:
                    env_data = env_data.repeat(1, 2, 1)
                    env_mask = env_mask.repeat(1, 1, 2)
                # print("test-----",policy_data.shape,env_data.shape,policy_mask.shape,env_mask.shape)
                emb_policy = policy_encoder.infer_posterior(policy_data.detach().to(device),
                                                            policy_mask.detach().to(device)).detach()
                emb_env = env_encoder.infer_posterior(env_data.detach().to(device),
                                                      env_mask.detach().to(device)).detach()
                if norm:
                    emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
                    emb_env = F.normalize(emb_env, p=2, dim=1).detach()
                pred_value = value_net(torch.FloatTensor(init_obs_list[pi][env][index]).unsqueeze(0).to(device),
                                       emb_env.to(device), emb_policy.to(device)).item()

                acc += abs(pred_value - episode_reward_tensor)
                l2_loss = nn.MSELoss()
                loss = l2_loss(torch.tensor([pred_value]), torch.tensor([episode_reward_tensor]))
                acc_mse += loss.item()
                print(
                    f"Initial Policy: {pi} env: {env} --- norm init true reward: {episode_reward_tensor: .3f} --- predicted: {pred_value: .3f}")

            accurancy[env][pi] = acc / 10
            accurancy_mse[env][pi] = acc_mse / 10
    train_acc_mse = 0
    train_acc = 0
    test_acc1 = 0
    test_acc2 = 0
    for i in train_envs:
        for j in train_policies:
            train_acc += accurancy[i][j]

    for i in train_envs:
        for j in eval_policies:
            test_acc1 += accurancy[i][j]
    for i in eval_envs:
        for j in train_policies:
            test_acc2 += accurancy[i][j]

    test_acc = np.sum(accurancy) - train_acc - test_acc1 - test_acc2
    for i in train_envs:
        for j in train_policies:
            train_acc_mse += accurancy_mse[i][j]
    test_acc_mse = np.sum(accurancy_mse) - train_acc_mse
    print("accurancy matrix", accurancy, np.mean(accurancy), np.mean(accurancy, axis=0), np.mean(accurancy, axis=1),
          train_acc / 225, test_acc1 / 75, test_acc2 / 75, test_acc / 25)
    print(accurancy_mse, train_acc_mse / 225, test_acc_mse / 175)
    np.save("result/value_function_accurate_minitest_pdvf", accurancy)
    print("-------------------eval-------------------")

    seed_list = [2, 0, 2, 1, 4, 0, 4, 0, 2, 3, 2, 1, 3, 1, 4, 1, 3, 2, 4, 0]  # ds0
    names = []
    for e in range(args.num_envs):
        names.append('ppo.{}.env{}.seed{}.pt'.format(args.env_name, e, int(seed_list[e])))
    all_policies = []
    for name in names:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)
        model = os.path.join(args.save_dir, name)
        actor_critic.load_state_dict(torch.load(model))
        all_policies.append(actor_critic)

    env_sampler = env_utils.EnvSamplerPDVF(envs, all_policies, args)
    # Eval on Train Envs
    policy_encoder.decoder.eval()
    value_net.eval()
    train_rewards = {}
    unnorm_train_rewards = {}
    unnorm_init_episode_reward_list = {}
    unnorm_decoder_reward_list = {}
    args.gamma = 1.0
    for ei in range(len(all_envs)):
        train_rewards[ei] = []
        unnorm_train_rewards[ei] = []
        unnorm_init_episode_reward_list[ei] = []
        unnorm_decoder_reward_list[ei] = []

    for i in range(NUM_EVAL_EPS):
        for ei in train_envs:
            for pi in train_policies:
                init_obs = torch.FloatTensor(env_sampler.env.reset(env_id=ei))
                # print(args.env_name)
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    init_state = env_sampler.env.sim.get_state()
                    res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=pi,
                                                                                env_idx=ei)
                else:
                    init_state = env_sampler.env.state
                    res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=pi,
                                                                         env_idx=ei)

                source_env = res['source_env']
                mask_env = res['mask_env']
                source_policy = res['source_policy']
                init_episode_reward = res['episode_reward']
                mask_policy = res['mask_policy']
                # print(source_policy.shape, mask_policy.shape, source_env.shape, mask_env.shape)
                if source_policy.shape[1] == 1:
                    source_policy = source_policy.repeat(1, 2, 1)
                    mask_policy = mask_policy.repeat(1, 1, 2)
                emb_policy = policy_encoder.infer_posterior(source_policy.detach().to(device),
                                                            mask_policy.detach().to(device)).detach()
                if source_env.shape[1] == 1:
                    source_env = source_env.repeat(1, 2, 1)
                    mask_env = mask_env.repeat(1, 1, 2)
                emb_env = env_encoder.infer_posterior(source_env.detach().to(device),
                                                      mask_env.detach().to(device)).detach()

                emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
                emb_env = F.normalize(emb_env, p=2, dim=1).detach()
                # emb_input = torch.FloatTensor(emb_policy)
                # print(emb_input)
                # emb_input.require_grad = True
                emb_input = torch.nn.Parameter(emb_policy, requires_grad=True)
                target = -value_net(init_obs.unsqueeze(0).to(device).detach(),
                                    emb_env.detach(), emb_input).mean()
                # print(target)
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state_mujoco(args,
                                                                                       init_state, init_obs, emb_policy,
                                                                                       policy_encoder.decoder,
                                                                                       env_idx=ei)[0]
                else:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state(args,
                                                                                init_state, init_obs, emb_policy,
                                                                                policy_encoder.decoder, env_idx=ei)[0]

                # pred_value = target.item()
                target.backward()
                # print(emb_input.grad)
                emb_policy_after_optimal = emb_input + emb_input.grad * args.op_lr

                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state_mujoco(
                        args, init_state, init_obs, emb_policy_after_optimal, policy_encoder.decoder, env_idx=ei)
                else:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state(
                        args, init_state, init_obs, emb_policy_after_optimal, policy_encoder.decoder, env_idx=ei)

                episode_reward = episode_reward_pos
                opt_policy = emb_policy_after_optimal

                unnorm_episode_reward = episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_init_episode_reward = init_episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_decoded_reward = decoded_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_init_episode_reward_list[ei].append(unnorm_init_episode_reward)
                unnorm_train_rewards[ei].append(unnorm_episode_reward)
                train_rewards[ei].append(episode_reward)
                unnorm_decoder_reward_list[ei].append(unnorm_decoded_reward)
                if i % args.log_interval == 0:
                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        print(
                            f"\nTrain Environemnt: {ei}--- reward after update: {unnorm_episode_reward: .3f}")
                        print(
                            f"Initial Policy: {pi} --- init true reward: {unnorm_init_episode_reward: .3f} --- decoded: {unnorm_decoded_reward: .3f} ")  # --- predicted: {pred_value: .3f}
                    print(
                        f"Train Environemnt: {ei} --- norm reward after update: {episode_reward: .3f}")
                    print(
                        f"Initial Policy: {pi} --- norm init true reward: {init_episode_reward: .3f} --- norm decoded: {decoded_reward: .3f} ")  # --- predicted: {pred_value: .3f}
    print(unnorm_init_episode_reward_list)
    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(unnorm_train_rewards[ei]), np.std(unnorm_train_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(train_rewards[ei]), np.std(train_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))
    # Eval on Eval Envs
    eval_rewards = {}
    unnorm_eval_rewards = {}
    unnorm_init_episode_reward = {}
    unnorm_decoder_reward_list = {}
    for ei in range(len(all_envs)):
        eval_rewards[ei] = []
        unnorm_eval_rewards[ei] = []
        unnorm_init_episode_reward_list[ei] = []
        unnorm_decoder_reward_list[ei] = []
    decoded_reward_process = np.zeros((len(eval_envs), len(train_policies), args.gd_iter))

    for ei in eval_envs:
        for pi in train_policies:
            init_obs = torch.FloatTensor(env_sampler.env.reset(env_id=ei))

            if 'ant' in args.env_name or 'swimmer' in args.env_name:
                init_state = env_sampler.env.sim.get_state()
                res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=pi,
                                                                            env_idx=ei)
            else:
                init_state = env_sampler.env.state
                res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=pi,
                                                                     env_idx=ei)

            source_env = res['source_env']
            mask_env = res['mask_env']
            source_policy = res['source_policy']
            init_episode_reward = res['episode_reward']
            mask_policy = res['mask_policy']

            if source_policy.shape[1] == 1:
                source_policy = source_policy.repeat(1, 2, 1)
                mask_policy = mask_policy.repeat(1, 1, 2)
            emb_policy = policy_encoder.infer_posterior(source_policy.detach().to(device),
                                                        mask_policy.detach().to(device)).detach()

            if source_env.shape[1] == 1:
                source_env = source_env.repeat(1, 2, 1)
                mask_env = mask_env.repeat(1, 1, 2)
            emb_env = env_encoder.infer_posterior(source_env.detach().to(device),
                                                  mask_env.detach().to(device)).detach()

            emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
            emb_env = F.normalize(emb_env, p=2, dim=1).detach()
            for j in range(args.gd_iter):

                emb_input = torch.nn.Parameter(emb_policy, requires_grad=True)
                target = -value_net(init_obs.unsqueeze(0).to(device).detach(),
                                    emb_env.detach(), emb_input).mean()
                # print(target)
                decoded_reward = 0
                for i in range(NUM_EVAL_EPS):

                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        decoded_reward += env_sampler.get_reward_pol_embedding_state_mujoco(args,
                                                                                            init_state, init_obs,
                                                                                            emb_policy,
                                                                                            policy_encoder.decoder,
                                                                                            env_idx=ei)[0]
                    else:
                        decoded_reward += env_sampler.get_reward_pol_embedding_state(args,
                                                                                     init_state, init_obs,
                                                                                     emb_policy,
                                                                                     policy_encoder.decoder,
                                                                                     env_idx=ei)[0]
                decoded_reward = decoded_reward / NUM_EVAL_EPS
                decoded_reward_process[ei - 15][pi][j] = decoded_reward
                # pred_value = target.item()
                target.backward()
                # print(emb_input.grad)
                emb_policy_after_optimal = emb_input + emb_input.grad * args.op_lr
                emb_policy = emb_policy_after_optimal.detach()

            max_reward_index = np.argmax(decoded_reward_process[ei - 15][pi])
            # print(max_reward_index)
            episode_reward = decoded_reward_process[ei - 15][pi][max_reward_index]
            # print(episode_reward)
            opt_policy = emb_policy_after_optimal
            print(episode_reward)
            unnorm_episode_reward = episode_reward * (args.max_reward - args.min_reward) + args.min_reward
            unnorm_init_episode_reward = init_episode_reward * (args.max_reward - args.min_reward) + args.min_reward
            unnorm_decoded_reward = decoded_reward * (args.max_reward - args.min_reward) + args.min_reward
            unnorm_init_episode_reward_list[ei].append(unnorm_init_episode_reward)
            unnorm_eval_rewards[ei].append(unnorm_episode_reward)
            eval_rewards[ei].append(episode_reward)
            unnorm_decoder_reward_list[ei].append(unnorm_decoded_reward)
            if i % args.log_interval == 0:
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    print(
                        f"\nEval Environemnt: {ei}  ")
                    print(
                        f"Initial Policy: {pi} --- init true reward: {unnorm_init_episode_reward: .3f} --- decoded: {unnorm_decoded_reward: .3f} --- reward after update: {unnorm_episode_reward: .3f}")
                print(
                    f"Eval Environemnt: {ei}  --- norm reward after update: {episode_reward: .3f}")
                print(
                    f"Initial Policy: {pi} --- norm init true reward: {init_episode_reward: .3f} --- norm decoded: {decoded_reward: .3f} ")

    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(unnorm_train_rewards[ei]), np.std(unnorm_train_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(train_rewards[ei]), np.std(train_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))

    for ei in eval_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Eval Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(unnorm_eval_rewards[ei]), np.std(unnorm_eval_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))
        else:
            print("Eval Env {} has reward with mean {:.3f} and std {:.3f}" \
                  .format(ei, np.mean(eval_rewards[ei]), np.std(eval_rewards[ei])))
            print("Train Env {} has init reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_init_episode_reward_list[ei])))
            print("Train Env {} has decode reward with mean {:.3f}" \
                  .format(ei, np.mean(unnorm_decoder_reward_list[ei])))

    print(unnorm_init_episode_reward_list)
    reward_list = [[], [], [], [], []]
    for i in range(NUM_EVAL_EPS):
        for ei in eval_envs:
            init_obs = torch.FloatTensor(env_sampler.env.reset(env_id=ei))
            
            if 'ant' in args.env_name or 'swimmer' in args.env_name:
                init_state = env_sampler.env.sim.get_state()
                res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=ei,
                                                                            env_idx=ei)
            else:
                init_state = env_sampler.env.state
                res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=ei,
                                                                     env_idx=ei)

            init_episode_reward = res['episode_reward']
            reward_list[ei - 15].append(init_episode_reward)
    for ei in eval_envs:
        print(np.mean(np.array(reward_list[ei - 15])))
    envs.close()

if __name__ == '__main__':

    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # loaded = np.load("np0-total-swim-100-max-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    loaded = np.load("np0-total-ant-100-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np1-total-space-100-max-sas-maskenv-sa-maskpi.npz", allow_pickle=True)

    args = get_args()
    for i in range(3):
        # for i in range(5,10):

        args.num_epochs_emb_policy = 3000
        args.num_epochs_emb_env = 3000
        args.num_epochs_pdvf_phase1 = 3000
        args.num_epochs_emb_z = 1500
        args.seed = int(i+12)
        args.shuffle = 0
        args.data_set = 0
        args.num_eval_eps = 3
        
        args.use_information_bottleneck = False
        train_pdvf_ds(a, a, loaded, args)
