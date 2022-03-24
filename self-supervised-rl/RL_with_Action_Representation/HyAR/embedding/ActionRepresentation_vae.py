# TODO: s discrete continue ->s"
import numpy as np
import torch
from torch import float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from embedding.Utils.utils import NeuralNet, pairwise_distances, pairwise_hyp_distances, squash, atanh
from embedding.Utils import Basis
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as functional


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, action_embedding_dim, parameter_action_dim, latent_dim, max_action,
                 hidden_size=256):
        super(VAE, self).__init__()

        # embedding table
        init_tensor = torch.rand(action_dim,
                                 action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        print("self.embeddings", self.embeddings)
        self.e0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.e0_1 = nn.Linear(parameter_action_dim, hidden_size)

        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.d0_1 = nn.Linear(latent_dim, hidden_size)
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.parameter_action_output = nn.Linear(hidden_size, parameter_action_dim)

        self.d3 = nn.Linear(hidden_size, hidden_size)

        self.delta_state_output = nn.Linear(hidden_size, state_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action, action_parameter):

        z_0 = F.relu(self.e0_0(torch.cat([state, action], 1)))
        z_1 = F.relu(self.e0_1(action_parameter))
        z = z_0 * z_1

        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)

        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u, s = self.decode(state, z, action)

        return u, s, mean, std

    def decode(self, state, z=None, action=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        v_0 = F.relu(self.d0_0(torch.cat([state, action], 1)))
        v_1 = F.relu(self.d0_1(z))
        v = v_0 * v_1
        v = F.relu(self.d1(v))
        v = F.relu(self.d2(v))

        parameter_action = self.parameter_action_output(v)

        v = F.relu(self.d3(v))
        s = self.delta_state_output(v)

        if raw: return parameter_action, s
        return self.max_action * torch.tanh(parameter_action), torch.tanh(s)


class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 parameter_action_dim,
                 reduced_action_dim=2,
                 reduce_parameter_action_dim=2,
                 embed_lr=1e-4,
                 ):
        super(Action_representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduce_parameter_action_dim = reduce_parameter_action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        self.latent_dim = self.reduce_parameter_action_dim
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=self.state_dim, action_dim=self.action_dim,
                       action_embedding_dim=self.reduced_action_dim, parameter_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, max_action=1.0,
                       hidden_size=256).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

    def discrete_embedding(self, ):
        emb = self.vae.embeddings

        return emb

    def unsupervised_loss(self, s1, a1, a2, s2, sup_batch_size, embed_lr):

        a1 = self.get_embedding(a1).to(self.device)

        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)

        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a1, a2, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):

        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True)
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        vae_loss = recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss
        # print("vae loss",vae_loss)
        # return vae_loss, 0.25 * recon_loss_s, recon_loss_c, 0.5 * KL_loss
        # return vae_loss, 0.25 * recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss #best
        return vae_loss, recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss

    def train_step(self, s1, a1, a2, s2, sup_batch_size, embed_lr=1e-4):
        state = s1
        action_d = a1
        action_c = a2
        next_state = s2
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,
                                                                  sup_batch_size)

        # 更新VAE
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def select_parameter_action(self, state, z, action):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
            action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
            action_c, state = self.vae.decode(state, z, action)
        return action_c.cpu().data.numpy().flatten()

    # def select_delta_state(self, state, z, action):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #         z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
    #         action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #         action_c, state = self.vae.decode(state, z, action)
    #     return state.cpu().data.numpy().flatten()
    def select_delta_state(self, state, z, action):
        with torch.no_grad():
            action_c, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.vae.embeddings[action]
        action_emb = torch.tanh(action_emb)
        return action_emb

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.vae.embeddings
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        return similarity

        # 获得最优动作，输出于embedding最相近的action 作为最优动作.

    def select_discrete_action(self, action):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        # print("pos",pos,len(pos))
        if len(pos) == 1:
            return pos.cpu().item()  # data.numpy()[0]
        else:
            # print("pos.cpu().item()", pos.cpu().numpy())
            return pos.cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
        # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)

    def get_c_rate(self, s1, a1, a2, s2, batch_size=100, range_rate=5):
        a1 = self.get_embedding(a1).to(self.device)
        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)
        recon_c, recon_s, mean, std = self.vae(s1, a1, a2)
        # print("recon_s",recon_s.shape)
        z = mean + std * torch.randn_like(std)
        z = z.cpu().data.numpy()
        c_rate = self.z_range(z, batch_size, range_rate)
        # print("s2",s2.shape)

        recon_s_loss = F.mse_loss(recon_s, s2, size_average=True)

        # recon_s = abs(np.mean(recon_s.cpu().data.numpy()))
        return c_rate, recon_s_loss.detach().cpu().numpy()

    def z_range(self, z, batch_size=100, range_rate=5):

        self.z1, self.z2, self.z3, self.z4, self.z5, self.z6, self.z7, self.z8, self.z9,\
        self.z10,self.z11,self.z12,self.z13,self.z14,self.z15,self.z16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        border = int(range_rate * (batch_size / 100))

        # print("border",border)
        if len(z[0]) == 16:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])
                self.z11.append(z[i][10])
                self.z12.append(z[i][11])
                self.z13.append(z[i][12])
                self.z14.append(z[i][13])
                self.z15.append(z[i][14])
                self.z16.append(z[i][15])

        if len(z[0]) == 16:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort(),self.z13.sort(), self.z14.sort(), self.z15.sort(), self.z16.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_13_up = self.z13[-border - 1]
            c_rate_13_down = self.z13[border]
            c_rate_14_up = self.z14[-border - 1]
            c_rate_14_down = self.z14[border]
            c_rate_15_up = self.z15[-border - 1]
            c_rate_15_down = self.z15[border]
            c_rate_16_up = self.z16[-border - 1]
            c_rate_16_down = self.z16[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, \
            c_rate_9, c_rate_10, c_rate_11, c_rate_12, c_rate_13, c_rate_14, c_rate_15, c_rate_16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            c_rate_13.append(c_rate_13_up), c_rate_13.append(c_rate_13_down)
            c_rate_14.append(c_rate_14_up), c_rate_14.append(c_rate_14_down)
            c_rate_15.append(c_rate_15_up), c_rate_15.append(c_rate_15_down)
            c_rate_16.append(c_rate_16_up), c_rate_16.append(c_rate_16_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8,\
                   c_rate_9, c_rate_10, c_rate_11, c_rate_12,c_rate_13, c_rate_14, c_rate_15, c_rate_16

        if len(z[0]) == 12:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])
                self.z11.append(z[i][10])
                self.z12.append(z[i][11])

        if len(z[0]) == 12:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12 = [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12

        if len(z[0]) == 10:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])

        if len(z[0]) == 10:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), self.z9.sort(), self.z10.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10 = [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10

        if len(z[0]) == 8:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])

        if len(z[0]) == 8:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8 = [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8

        if len(z[0]) == 6:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])

        if len(z[0]) == 6:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6 = [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6

        if len(z[0]) == 4:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])

        if len(z[0]) == 4:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4 = [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4

        if len(z[0]) == 3:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])

        if len(z[0]) == 3:
            self.z1.sort(), self.z2.sort(), self.z3.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]

            c_rate_1, c_rate_2, c_rate_3 = [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)

            return c_rate_1, c_rate_2, c_rate_3
