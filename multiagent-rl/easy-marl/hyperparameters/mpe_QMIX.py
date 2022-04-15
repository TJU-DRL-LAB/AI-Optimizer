class Hyperparameter(object):
    def __init__(self):
        # basic hyperparameters
        self.agent_name = 'QMIX'
        self.env_name = 'MPE'  ## 需要具体的场景应该
        self.exp_name = 'MPE_IDQN'
        self.alg_type = 'DQN_based'
        # self.con_or_dis =
        ## todo
        # 需要的参数包括：离散还是连续？有什么区别？MPE中具体场景的名称

        # control flow hyperparameters
        self.exp_count = 1
        self.episode_count = 500
        self.episode_max_step = 40
        self.buffer_type = "step"
        self.evaluate_interval = 100
        self.evaluate_episode_num = 100
        self.bool_evaluate = False
        self.epsilon = 0.5
        self.min_epsilon = 0
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / 1000
        self.train_interval = 1
        self.save_interval = 100
        self.model_dir = "./model/" + (self.agent_name + self.env_name) + "/"

        # agent-dnn hyperparameters
        self.hidden_dim = 32
        self.hidden_layer_count = 2

        # agent-algorithm hyperparameters
        self.lr = 0.001
        self.gamma = 0.9
        self.clip_grad_norm = 10.0

        # env hyperparameters
        # todo 添加MPE具体场景的相应参数，问题在于目前MPE环境中的参数是通过yaml事先定义好的，如何在这里进行设置？直接读还是？
        # 区分一下哪些参数需要设定，那些参数是直接读的
        # 需要包括的参数有：agent数目，最大步数，ob_dim和a_dim（这个应该是从环境中获取)，所以这个文件里面就要读环境中的东西吗？
        # 需要设定的参数
        self.agent_count = 3
        # 环境内置的参数（与算法相关），应该直接读或者直接固定死就行
        self.observation_dim_list = [18, 18, 18]
        self.action_dim_list = [5, 5, 5]

        # replay buffer hyperparameters
        self.buffer_size = 1e6
        self.batch_size = 64

