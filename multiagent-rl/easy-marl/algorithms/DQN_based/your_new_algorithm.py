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


# define your dnn structure
class YourDnnStructure(nn.Module):
    def __init__(self, args):
        super(YourDnnStructure, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        # define your dnn parameters
        raise NotImplementedError()

    def forward(self, observation_list):
        # define your dnn forward method (to generate Q-values or actions)
        raise NotImplementedError()

    def your_special_method(self):
        # define your special method to generate special values (e.g., q_total)
        raise NotImplementedError()


# define your DRL algorithm
class YourAlgorithm(object):
    def __init__(self, args):
        self.args = args
        self.M_net = YourDnnStructure(args)  # main network
        self.T_net = YourDnnStructure(args)  # target network
        self._init_necessary_info()

    # you may possibly change this method for your purpose
    def generate_q_list(self, observation_list):
        self._set_evaluation_mode()

        observation_list = [to_tensor(observation) for observation in observation_list]
        q_list = self.M_net(observation_list)
        return [q_value.detach().cpu().numpy()[0] for q_value in q_list]

    # you may possibly change this method for your purpose
    def train(self):
        self._set_train_mode()

        # write the training logic for your DRL algorithm
        raise NotImplementedError()

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

