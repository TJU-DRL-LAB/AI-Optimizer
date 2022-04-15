class Hyperparameter(object):
    def __init__(self, scenario):
        # basic hyperparameters
        self.agent_name = 'IDQN'
        self.env_name = 'discrete_magym'
        self.scenario_name = 'Switch4-v0' if (scenario=="" or scenario=="None") else scenario
        # self.alg_type = 'DQN_based'

        # control flow hyperparameters
        self.exp_count = 1
        self.episode_count = 200
        self.episode_max_step = 20
        self.test_interval = 100
        self.test_episode_count = 100
        self.epsilon = 1
        self.min_epsilon = 0
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / 1000
        self.train_interval = 1
        self.save_interval = 100

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

