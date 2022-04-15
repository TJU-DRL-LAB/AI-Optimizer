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


class ActorDNN(nn.Module):
    def __init__(self, args):
        super(ActorDNN, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()   # do not use python dict {}
            # parameters for pre-processing observations
            parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            for j in range(self.args.hidden_layer_count):
                parameters_dict["fc_hidden" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating Qvalues
            parameters_dict["action"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[i])
            self.parameters_all_agent.append(parameters_dict)

    def forward(self, observation_list):
        action_list = []
        for i in range(self.args.agent_count):
            hidden = F.relu(self.parameters_all_agent[i]["fc_obs"](observation_list[i]))

            for j in range(self.args.hidden_layer_count):
                hidden = F.relu(self.parameters_all_agent[i]["fc_hidden" + str(j)](hidden))

            action = F.tanh(self.parameters_all_agent[i]["action"](hidden))  # tanh activation for action
            action_list.append(action)
        return action_list


class CriticDNN(nn.Module):
    def __init__(self, args):
        super(CriticDNN, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()  # do not use python dict {}
            # parameters for pre-processing observations and actions
            parameters_dict["fc_obs_action"] = nn.Linear(self.args.observation_dim_list[i] + self.args.action_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            for j in range(self.args.hidden_layer_count):
                parameters_dict["fc_hidden" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating q_value
            parameters_dict["q_value"] = nn.Linear(self.args.hidden_dim, 1)
            self.parameters_all_agent.append(parameters_dict)

    def forward(self, observation_list, action_list):
        q_list = []
        for i in range(self.args.agent_count):
            obs_action_cat = torch.cat([observation_list[i], action_list[i]], dim=-1)
            hidden = F.relu(self.parameters_all_agent[i]["fc_obs_action"](obs_action_cat))

            for j in range(self.args.hidden_layer_count):
                hidden = F.relu(self.parameters_all_agent[i]["fc_hidden" + str(j)](hidden))

            q_value = self.parameters_all_agent[i]["q_value"](hidden)  # linear activation for Q-value
            q_list.append(q_value)
        return q_list


class IDDPG(object):
    def __init__(self, args):
        self.args = args
        self.M_actor = ActorDNN(args)  # main network
        self.T_actor = ActorDNN(args)  # target network
        self.M_critic = CriticDNN(args)  # main network
        self.T_critic = CriticDNN(args)  # target network
        self._init_necessary_info()

    def generate_action(self, observation_list):
        self._set_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_list = self.M_actor(observation_list)
        return [action.detach().cpu().numpy() for action in action_list]

    def train(self, batch_experience_dict):
        self._set_train_mode()

        # prepare the training data
        observation_list = batch_experience_dict["agent_specific"]["observation_list"]
        continuous_action_list = batch_experience_dict["agent_specific"]["continuous_action_list"]
        reward_list = batch_experience_dict["agent_specific"]["reward_list"]
        next_observation_list = batch_experience_dict["agent_specific"]["next_observation_list"]
        multiplier = 1.0 - batch_experience_dict["shared"]["done"]

        # the following is the formal training logic for critic
        M_q_list = self.M_critic(observation_list, continuous_action_list)  # use M_net
        next_action_list = self.T_actor(next_observation_list)  # use T_net
        T_q_list = self.T_critic(next_observation_list, next_action_list)  # use T_net

        critic_loss = 0.0
        for i in range(self.args.agent_count):
            TD_target = reward_list[i] + multiplier * self.args.gamma * T_q_list[i]
            critic_loss += self.MSEloss(M_q_list[i], TD_target.detach())  # note the detach

        # optimize critic
        self.optimizer_critic.zero_grad()  # clear previous gradients before update
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.M_critic.parameters(), self.args.clip_grad_norm)  # after backward() before step()
        self.optimizer_critic.step()

        # the following is the formal training logic for actor
        policy_action_list = self.M_actor(observation_list)  # use M_net
        policy_q_list = self.M_critic(observation_list, policy_action_list)  # use M_net
        actor_loss = 0.0
        for i in range(self.args.agent_count):
            actor_loss += -policy_q_list[i].mean()  # minimize (-Q) -> maximize Q

        # optimize actor
        self.optimizer_actor.zero_grad()  # clear previous gradients before update
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.M_actor.parameters(),
                                 self.args.clip_grad_norm)  # after backward() before step()
        self.optimizer_actor.step()

        # Soft update target net
        self._train_target_network_soft()

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.M_actor.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
        for m in self.M_critic.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # init target network before training
        self._train_target_network_hard()

        # set target network to evaluation mode
        self.T_actor.eval()
        self.T_critic.eval()

        # create optimizers
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.optimizer_actor = torch.optim.Adam(self.M_actor.parameters(), lr=self.args.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.M_critic.parameters(), lr=self.args.lr_critic)

        if USE_CUDA:
            self._config_cuda()

    def _train_target_network_soft(self):
        for T_param, M_param in zip(self.T_actor.parameters(), self.M_actor.parameters()):
            T_param.data.copy_(T_param.data * (1.0 - self.args.tau) + M_param.data * self.args.tau)
        for T_param, M_param in zip(self.T_critic.parameters(), self.M_critic.parameters()):
            T_param.data.copy_(T_param.data * (1.0 - self.args.tau) + M_param.data * self.args.tau)

    def _train_target_network_hard(self):
        for T_param, M_param in zip(self.T_actor.parameters(), self.M_actor.parameters()):
            T_param.data.copy_(M_param.data)
        for T_param, M_param in zip(self.T_critic.parameters(), self.M_critic.parameters()):
            T_param.data.copy_(M_param.data)

    def _config_cuda(self):
        self.M_actor.cuda()
        self.T_actor.cuda()
        self.M_critic.cuda()
        self.T_critic.cuda()

    def _set_train_mode(self):
        self.M_actor.train()  # set train mode
        self.M_critic.train()

    def _set_evaluation_mode(self):
        self.M_actor.eval()  # set evaluation mode
        self.M_critic.eval()

    def save_model(self, model_dir):
        print("save_model() ...")
        torch.save(self.M_actor.state_dict(), '{}-actor.pkl'.format(model_dir))
        torch.save(self.M_critic.state_dict(), '{}-critic.pkl'.format(model_dir))

    def load_weights(self, model_dir):
        print("load_weights() ...")
        self.M_actor.load_state_dict(torch.load('{}-actor.pkl'.format(model_dir)))
        self.M_critic.load_state_dict(torch.load('{}-critic.pkl'.format(model_dir)))
        self._train_target_network_hard()

