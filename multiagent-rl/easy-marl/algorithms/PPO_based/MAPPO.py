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


class ActorDNN(nn.Module):
    def __init__(self, args):
        super(ActorDNN, self).__init__()
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

            # parameters for generating logit
            parameters_dict["action_logit_value"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[i])
            self.parameters_all_agent.append(parameters_dict)

    def foward_agent_i(self, observation, agent_id):
        hidden = F.relu(self.parameters_all_agent[agent_id]["fc_obs"](observation))

        for j in range(self.args.hidden_layer_count):
            hidden = F.relu(self.parameters_all_agent[agent_id]["fc_hidden" + str(j)](hidden))

        action_logit_value = self.parameters_all_agent[agent_id]["action_logit_value"](hidden)  # linear activation for Q-value

        return action_logit_value

    def forward(self, observation_list):
        action_logit_list = []

        for i, observation in enumerate(observation_list):
            action_logit_value = self.foward_agent_i(observation, i)
            action_logit_list.append(action_logit_value)
           
        return action_logit_list


class CriticDNN(nn.Module):
    def __init__(self, args):
        super(CriticDNN, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()  # do not use python dict {}
            # parameters for pre-processing state and actions
            parameters_dict["fc_state"] = nn.Linear(self.args.state_dim, self.args.hidden_dim)

            # parameters for hidden layers
            for j in range(self.args.hidden_layer_count):
                parameters_dict["fc_hidden" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating state value
            parameters_dict["state_value"] = nn.Linear(self.args.hidden_dim, 1)
            self.parameters_all_agent.append(parameters_dict)

    def foward_agent_i(self, state, agent_id):
        hidden = F.relu(self.parameters_all_agent[agent_id]["fc_state"](state))

        for j in range(self.args.hidden_layer_count):
            hidden = F.relu(self.parameters_all_agent[agent_id]["fc_hidden" + str(j)](hidden))

        state_value = self.parameters_all_agent[agent_id]["state_value"](hidden)

        return state_value


    def forward(self, state):
        state_value_list = []

        for i in range(self.args.agent_count):
            state_value = self.foward_agent_i(state, i)
            state_value_list.append(state_value)

        return state_value_list


class MAPPO(object):
    def __init__(self, args):
        self.args = args
        self.actor = ActorDNN(args)
        self.critic = CriticDNN(args)
        self._init_necessary_info()

    def generate_action_list(self, observation_list):
        self._set_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_logit_list = self.actor(observation_list)
        action_prob_list = [F.softmax(action_logit_value, dim=-1) for action_logit_value in action_logit_list]
        action_list = [Categorical(action_prob).sample().item() for action_prob in action_prob_list]
        action_prob_list = [action_prob[..., action_id].detach().cpu().numpy() for action_prob, action_id in zip(action_prob_list, action_list)]
        return action_list, action_prob_list

    def generate_action_prob_V_agent_i(self, observation, agent_id, state):
        action_logit = self.actor.foward_agent_i(observation, agent_id)
        action_prob = F.softmax(action_logit, dim=-1)
        state_value = self.critic.foward_agent_i(state, agent_id)
        return action_prob, state_value

    def train(self, batch):
        self._set_train_mode()
        total_actor_loss = 0
        total_critic_loss = 0

        for i in range(self.args.agent_count):
            observation_batch = batch["agent_specific"]["observation_list"][i]
            action_id_batch = batch["agent_specific"]["action_id_list"][i]
            reward_batch = batch["agent_specific"]["reward_list"][i].reshape(-1, 1)
            next_observation_batch = batch["agent_specific"]["next_observation_list"][i]
            action_prob_batch = batch["agent_specific"]["action_prob_list"][i]
            state_batch = batch["shared"]["state"]
            
            done = batch["shared"]["done"].reshape(-1, 1)
            multiplier = 1.0 - done

            # the following is the formal training logic
            new_action_prob, state_value = self.generate_action_prob_V_agent_i(observation_batch, i, state_batch)
            _, next_state_value = self.generate_action_prob_V_agent_i(next_observation_batch, i, state_batch)


            state_value_target = reward_batch + multiplier * self.args.gamma * next_state_value

            # for critic
            critic_loss = F.smooth_l1_loss(state_value, state_value_target.detach())
            total_critic_loss += critic_loss

            # for ppo
            action = action_id_batch.to(torch.int64)
            action_prob = new_action_prob.gather(-1, action).squeeze(-1)
            old_action_prob = action_prob_batch
            advantage = state_value_target - state_value
            ratio = torch.exp(torch.log(action_prob) - torch.log(old_action_prob))
            ppo_loss = -torch.mean(torch.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantage.detach())
            total_actor_loss += ppo_loss

            # for exploration
            entropy = Categorical(action_prob).entropy()  
            total_actor_loss -= entropy
           
        self.actor_optimizer.zero_grad()  # clear previous gradients before update
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clip_grad_norm)  # after backward() before step()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()  # clear previous gradients before update
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clip_grad_norm)  # after backward() before step()
        self.critic_optimizer.step()

        return total_actor_loss.detach().cpu().numpy() + total_critic_loss.detach().cpu().numpy()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.actor.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
        
        for m in self.critic.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)


        # create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        if USE_CUDA:
            self._config_cuda()

    def _config_cuda(self):
        self.actor.cuda()
        self.critic.cuda()

    def _set_train_mode(self):
        self.actor.train()  # set train mode
        self.critic.train()

    def _set_evaluation_mode(self):
        self.actor.eval()  # set evaluation mode
        self.critic.eval()

    def save_model(self, model_dir):
        print("save_model() ...")
        torch.save(self.actor.state_dict(), '{}-actor.pkl'.format(model_dir))
        torch.save(self.critic.state_dict(), '{}-critic.pkl'.format(model_dir))

    def load_weights(self, model_dir):
        print("load_weights() ...")
        self.actor.load_state_dict(torch.load('{}-actor.pkl'.format(model_dir)))
        self.critic.load_state_dict(torch.load('{}-critic.pkl'.format(model_dir)))

