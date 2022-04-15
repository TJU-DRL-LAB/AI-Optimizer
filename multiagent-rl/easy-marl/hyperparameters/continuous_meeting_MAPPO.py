class Hyperparameter(object):
    def __init__(self):
        # basic hyperparameters
        self.agent_name = 'MAPPO'
        self.env_name = 'continuous_meeting'
        # self.alg_type = 'PPO_based'

        # control flow hyperparameters
        self.exp_count = 1
        self.episode_count = 20
        self.episode_max_step = 20
        self.test_interval = 10
        self.test_episode_count = 10
        # self.train_interval = 1
        self.save_interval = 20

        # agent-dnn hyperparameters
        self.hidden_dim = 32
        self.hidden_layer_count = 2

        # agent-algorithm hyperparameters
        self.lr = 0.001
        self.gamma = 0.9
        self.clip_grad_norm = 10.0

        # replay buffer hyperparameters
        self.buffer_size = 1e6
        self.batch_size = 64

        # algo hyperparameter
        self.eps_clip = 0.1

