import gym
import torch
from core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet


class ClassicControlConfig(BaseMuZeroConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=20000,
            test_interval=100,
            test_episodes=5,
            checkpoint_interval=20,
            max_moves=1000,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=128,
            td_steps=5,
            num_actors=32,
            lr_init=0.05,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=1000,
            value_loss_coeff=1,
            value_support=DiscreteSupport(-20, 20),
            reward_support=DiscreteSupport(-5, 5))

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return ClassicControlWrapper(env, discount=self.discount, k=4)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


muzero_config = ClassicControlConfig()
