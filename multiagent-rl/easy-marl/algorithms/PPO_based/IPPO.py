import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
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
                parameters_dict["fc_hidden" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating logit and state value
            parameters_dict["action_logit_value"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[i])
            parameters_dict["state_value"] = nn.Linear(self.args.hidden_dim, 1)
            self.parameters_all_agent.append(parameters_dict)

    def foward_agent_i(self, observation, agent_id):
        hidden = F.relu(self.parameters_all_agent[agent_id]["fc_obs"](observation))

        for j in range(self.args.hidden_layer_count):
            hidden = F.relu(self.parameters_all_agent[agent_id]["fc_hidden" + str(j)](hidden))

        action_logit_value = self.parameters_all_agent[agent_id]["action_logit_value"](hidden)  # linear activation for Q-value
        state_value = self.parameters_all_agent[agent_id]["state_value"](hidden)

        return action_logit_value, state_value


    def forward(self, observation_list):
        action_logit_list = []
        state_value_list = []

        for i, observation in enumerate(observation_list):
            action_logit_value, state_value = self.foward_agent_i(observation, i)
            action_logit_list.append(action_logit_value)
            state_value_list.append(state_value)

        return action_logit_list, state_value_list


class IPPO(object):
    def __init__(self, args):
        self.args = args
        self.net = DNN(args)
        self._init_necessary_info()

    def generate_action_list(self, observation_list):
        self._set_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_logit_list, _ = self.net(observation_list)
        action_prob_list = [F.softmax(action_logit_value, dim=-1) for action_logit_value in action_logit_list]
        action_list = [Categorical(action_prob).sample().item() for action_prob in action_prob_list]
        action_prob_list = [action_prob[..., action_id].detach().cpu().numpy() for action_prob, action_id in zip(action_prob_list, action_list)]
        return action_list, action_prob_list

    def generate_action_prob_V_agent_i(self, observation, agent_id):
        action_logit, state_value = self.net.foward_agent_i(observation, agent_id)
        action_prob = F.softmax(action_logit, dim=-1)
        return action_prob, state_value

    def train(self, batch):
        self._set_train_mode()
        total_loss = 0

        for i in range(self.args.agent_count):
            observation_batch = batch["agent_specific"]["observation_list"][i]
            action_id_batch = batch["agent_specific"]["action_id_list"][i]
            reward_batch = batch["agent_specific"]["reward_list"][i].reshape(-1, 1)
            next_observation_batch = batch["agent_specific"]["next_observation_list"][i]

            action_prob_batch = batch["agent_specific"]["action_prob_list"][i]
            
            done = batch["shared"]["done"].reshape(-1, 1)
            multiplier = 1.0 - done

            # the following is the formal training logic
            new_action_prob, state_value = self.generate_action_prob_V_agent_i(observation_batch, i)
            _, next_state_value = self.generate_action_prob_V_agent_i(next_observation_batch, i)


            state_value_target = reward_batch + multiplier * self.args.gamma * next_state_value

            # for critic
            critic_loss = F.smooth_l1_loss(state_value, state_value_target.detach())
            total_loss += critic_loss

            # for ppo
            action = action_id_batch.to(torch.int64)
            action_prob = new_action_prob.gather(-1, action).squeeze(-1)
            old_action_prob = action_prob_batch
            advantage = state_value_target - state_value
            ratio = torch.exp(torch.log(action_prob) - torch.log(old_action_prob))
            ppo_loss = -torch.mean(torch.min(ratio * advantage, torch.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantage))
            total_loss += ppo_loss

            # for exploration
            entropy = Categorical(action_prob).entropy()  
            total_loss -= entropy
           
        self.optimizer.zero_grad()  # clear previous gradients before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad_norm)  # after backward() before step()
        self.optimizer.step()

        return total_loss.detach().cpu().numpy()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.net.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # create optimizers
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

        if USE_CUDA:
            self._config_cuda()

    def _config_cuda(self):
        self.net.cuda()

    def _set_train_mode(self):
        self.net.train()  # set train mode

    def _set_evaluation_mode(self):
        self.net.eval()  # set evaluation mode

    def save_model(self, model_dir):
        print("save_model() ...")
        torch.save(self.net.state_dict(), '{}-net.pkl'.format(model_dir))

    def load_weights(self, model_dir):
        print("load_weights() ...")
        self.net.load_state_dict(torch.load('{}-net.pkl'.format(model_dir)))

