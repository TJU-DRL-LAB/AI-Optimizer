import os
import torch
from .game import Game


class DiscreteSupport:
    def __init__(self, min: int, max: int):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)


class BaseMuZeroConfig(object):

    def __init__(self,
                 training_steps: int,
                 test_interval: int,
                 test_episodes: int,
                 checkpoint_interval: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_rate: float,
                 lr_decay_steps: float,
                 window_size: int = int(1e6),
                 value_loss_coeff: float = 1,
                 value_support: DiscreteSupport = None,
                 reward_support: DiscreteSupport = None):

        # Self-Play
        self.action_space_size = None
        self.num_actors = num_actors

        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount
        self.max_grad_norm = 5

        # testing arguments
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the environment, we can use them to
        # initialize the rescaling. This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.max_value_bound = None
        self.min_value_bound = None

        # Training
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.value_loss_coeff = value_loss_coeff
        self.device = 'cpu'
        self.exp_path = None  # experiment path
        self.debug = False
        self.model_path = None
        self.seed = None
        self.value_support = value_support
        self.reward_support = reward_support

        # optimization control
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        # replay buffer
        self.priority_prob_alpha = 1
        self.use_target_model = True
        self.revisit_policy_search_rate = 0
        self.use_max_priority = None

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        raise NotImplementedError

    def set_game(self, env_name):
        raise NotImplementedError

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None) -> Game:
        """ returns a new instance of the game"""
        raise NotImplementedError

    def get_uniform_network(self):
        raise NotImplementedError

    def scalar_loss(self, prediction, target):
        raise NotImplementedError

    @staticmethod
    def scalar_transform(x):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    def inverse_scalar_transform(self, logits, scalar_support):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        value_probs = torch.softmax(logits, dim=1)
        value_support = torch.ones(value_probs.shape)
        value_support[:, :] = torch.tensor([x for x in scalar_support.range])
        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True)

        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    @staticmethod
    def _phi(x, min, max, set_size: int):
        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = (x - x_low)
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min, x_low - min
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    def get_hparams(self):
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        self.set_game(args.env)
        self.seed = args.seed
        self.priority_prob_alpha = 1 if args.use_priority else 0
        self.use_target_model = args.use_target_model
        self.debug = args.debug
        self.device = args.device
        self.use_max_priority = (args.use_max_priority and args.use_priority)

        if args.value_loss_coeff is not None:
            self.value_loss_coeff = args.value_loss_coeff

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        self.exp_path = os.path.join(args.result_dir, args.case, args.env,
                                     'revisit_rate_{}'.format(self.revisit_policy_search_rate),
                                     'val_coeff_{}'.format(self.value_loss_coeff),
                                     'with_target' if self.use_target_model else 'no_target',
                                     'with_prio' if args.use_priority else 'no_prio',
                                     'max_prio' if self.use_max_priority else 'no_max_prio',
                                     'seed_{}'.format(self.seed))

        self.model_path = os.path.join(self.exp_path, 'model.p')
        return self.exp_path
