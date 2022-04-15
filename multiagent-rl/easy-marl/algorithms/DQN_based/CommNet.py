import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()  # do not use python dict {}
            # parameters for pre-processing observations and actions
            parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            for j in range(self.args.hidden_layer_count):
                parameters_dict["fc_H" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
                parameters_dict["fc_C" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating q_value
            parameters_dict["q_value"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[i])
            self.parameters_all_agent.append(parameters_dict)

    def forward(self, observation_list):
        # pre-process
        obs_out_list = []
        for i in range(self.args.agent_count):
            obs_out = F.relu(self.parameters_all_agent[i]["fc_obs"](observation_list[i]))
            obs_out_list.append(obs_out)

        # key part of difference MARL methods
        hidden_out_list = obs_out_list
        for layer_index in range(self.args.hidden_layer_count):
            hidden_out_list = self._mean_communication(hidden_out_list, layer_index)

        # post-process
        q_list = []
        for i in range(self.args.agent_count):
            q_value = self.parameters_all_agent[i]["q_value"](hidden_out_list[i])  # linear activation for Q-value
            q_list.append(q_value)

        return q_list

    def _mean_communication(self, hidden_list, layer_index):
        output_list = []
        hidden_list = torch.stack(hidden_list, dim=0)  # (agent_count, batch_size, hidden_dim)
        hidden_sum = torch.sum(hidden_list, dim=0, keepdims=False)  # (batch_size, hidden_dim)
        for i in range(self.args.agent_count):
            message_i = (hidden_sum - hidden_list[i]) / (self.args.agent_count - 1)  # mean message operation
            param_H = self.parameters_all_agent[i]["fc_H" + str(layer_index)]  # in fact, it is a layer, not parameter
            param_C = self.parameters_all_agent[i]["fc_C" + str(layer_index)]
            output_i = torch.tanh(param_H(hidden_list[i]) + param_C(message_i))
            output_list.append(output_i)
        return output_list


class CommNet(object):
    def __init__(self, args):
        self.args = args
        self.M_net = DNN(args)  # main network
        self.T_net = DNN(args)  # target network
        self._init_necessary_info()

    def generate_q_list(self, observation_list):
        self._set_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        q_list = self.M_net(observation_list)
        return [q_value.detach().cpu().numpy()[0] for q_value in q_list]

    def train(self, batch):
        self._set_train_mode()

        # prepare the training data
        observation_list = []
        # state_list = []
        action_id_list = []
        reward_list = []
        next_observation_list = []
        for i in range(self.args.agent_count):
            observation_batch = np.asarray([e[0][i] for e in batch])
            observation_list.append(to_tensor(observation_batch))
            # state_batch = np.asarray([e[1][i] for e in batch])
            # state_list.append(to_tensor(state_batch))
            action_id_batch = np.asarray([e[2][i] for e in batch]).reshape(-1, 1)
            action_id_list.append(to_tensor(action_id_batch))
            reward_batch = np.asarray([e[3][i] for e in batch]).reshape(-1, 1)
            reward_list.append(to_tensor(reward_batch))
            next_observation_batch = np.asarray([e[4][i] for e in batch])
            next_observation_list.append(to_tensor(next_observation_batch))
        done = np.asarray([e[5] for e in batch]).astype(int).reshape(-1, 1)
        multiplier = to_tensor(1.0 - done)

        # the following is the formal training logic
        chosen_q_list = []
        target_q_list = []
        M_q_list = self.M_net(observation_list)
        T_q_list = self.T_net(next_observation_list)  # use T_net

        for i in range(self.args.agent_count):
            one_hot_action = F.one_hot(action_id_list[i].to(torch.int64), num_classes=self.args.action_dim_list[i])
            # action_id_list[i] is with shape=[None, 1], one_hot_action.shape == (None, 1, self.args.action_dim),
            # so we need to squeeze the second-dim
            one_hot_action = torch.squeeze(one_hot_action, dim=1)  # removes dimensions of size 1
            chosen_q_value = torch.sum(M_q_list[i] * one_hot_action, dim=1, keepdim=True)
            target_q_value = torch.max(T_q_list[i], dim=1, keepdim=True)[0]  # [0]: only return max values
            chosen_q_list.append(chosen_q_value)
            target_q_list.append(target_q_value)

        total_loss = 0.0
        for i in range(self.args.agent_count):
            TD_target = reward_list[i] + multiplier * self.args.gamma * target_q_list[i]
            total_loss += self.MSEloss(chosen_q_list[i], TD_target.cpu().detach())  # note the detach

        self.optimizer.zero_grad()  # clear previous gradients before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.M_net.parameters(), self.args.clip_grad_norm)  # after backward() before step()
        self.optimizer.step()

        # TODO: if it is time to train target network
        self._train_target_network_hard()

        return total_loss.detach().cpu().numpy()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.M_net.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # init target network before training
        self._train_target_network_hard()

        # set target network to evaluation mode
        self.T_net.eval()

        # create optimizers
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.M_net.parameters(), lr=self.args.lr)

        if USE_CUDA:
            self._config_cuda()

    def _train_target_network_hard(self):
        for T_param, M_param in zip(self.T_net.parameters(), self.M_net.parameters()):
            T_param.data.copy_(M_param.data)

    def _train_target_network_soft(self):
        for T_param, M_param in zip(self.T_net.parameters(), self.M_net.parameters()):
            T_param.data.copy_(T_param.data * (1.0 - self.args.tau) + M_param.data * self.args.tau)

    def _config_cuda(self):
        self.M_net.cuda()
        self.T_net.cuda()

    def _set_train_mode(self):
        self.M_net.train()  # set train mode

    def _set_evaluation_mode(self):
        self.M_net.eval()  # set evaluation mode

    def save_model(self, model_dir):
        print("save_model() ...")
        torch.save(self.M_net.state_dict(), '{}-net.pkl'.format(model_dir))

    def load_weights(self, model_dir):
        print("load_weights() ...")
        self.M_net.load_state_dict(torch.load('{}-net.pkl'.format(model_dir)))
        self._train_target_network_hard()

