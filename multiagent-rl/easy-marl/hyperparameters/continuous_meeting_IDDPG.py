
class Hyperparameter(object):
    def __init__(self):
        # basic hyperparameters
        self.agent_name = 'IDDPG'
        self.env_name = 'continuous_meeting'
        # self.alg_type = 'DDPG_based'

        # control flow hyperparameters
        self.exp_count = 5
        self.episode_count = 10
        self.episode_max_step = 25
        self.test_interval = 5
        self.test_episode_count = 100
        self.train_interval = 1
        self.save_interval = 100

        # agent-dnn hyperparameters
        self.hidden_dim = 32
        self.hidden_layer_count = 2

        # agent-algorithm hyperparameters
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.gamma = 0.9
        self.clip_grad_norm = 10.0
        self.tau = 0.01
        self.exploration_var = 0.1

        # replay buffer hyperparameters
        self.buffer_size = 1e6
        self.batch_size = 64


