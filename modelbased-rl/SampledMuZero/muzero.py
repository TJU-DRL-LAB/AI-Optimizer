import copy
import importlib
import math
import os
import pickle
import sys
import time
from glob import glob

import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime

import models
import replay_buffer
import self_play
import shared_storage
import trainer


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

    """

    def __init__(self, game_name, config=None, args=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    setattr(self.config, param, value)
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        ray.init(num_gpus=args.num_gpus,
                 num_cpus=args.num_cpus,
                 object_store_memory=args.object_store_memory,
                 ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        # SharedStorage checkpoint
        self.checkpoint = {
            "env_step": 0,  # env interactive step
            "weights": None,  # network weight
            "optimizer_state": None,  # network load dict
            "total_reward": 0,  # test reward
            "episode_length": 0,  # test average length
            "mean_value": 0,  #  test average value
            "training_step": 0,  # training step
            "lr": 0,  # learning rate
            "total_loss": 0,  # total loss
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,  # played games from buffer
            "num_played_steps": 0,  # played steps from buffer
            "num_reanalysed_games": 0,  # if reanalyse
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)

        # Manage GPUs
        # todo

        # Initialize workers
        num_gpus_per_worker = 1

        # training
        self.training_worker = trainer.Trainer.options(
            num_cpus=0, num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        # shared storage
        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        # replay buffer
        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        # reanalyse
        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        # self play
        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint, self.Game, self.config, self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0, num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print("\nTraining...\n")

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary", self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "env_step",
            "total_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        # counter for compare
        counter = 0
        try:
            while info["training_step"] < self.config.training_steps:
                training_step = info["training_step"]
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward", info["total_reward"], training_step,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value", info["mean_value"], training_step,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length", info["episode_length"], training_step,
                )

                writer.add_scalar(
                    "2.Workers/1.Self_played_games", info["num_played_games"], training_step,
                )
                writer.add_scalar(
                    "2.Workers/2.Self_played_steps", info["num_played_steps"], training_step
                )
                writer.add_scalar(
                    "2.Workers/3.Reanalysed_games", info["num_reanalysed_games"], training_step,
                )
                writer.add_scalar(
                    "2.Workers/4.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    training_step,
                )
                writer.add_scalar("2.Workers/5.Learning_rate", info["lr"], training_step)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], training_step
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], training_step)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], training_step)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], training_step)
                writer.add_scalar("4.Env step reward/Reward", info["total_reward"], info["env_step"])
                writer.add_scalar("5.Counter/Reward", info["total_reward"], counter)
                print(
                    f' Last test reward: {info["total_reward"]:.2f}.'
                    f' Training step: {info["training_step"]}/{self.config.training_steps}.'
                    f' Played games: {info["num_played_games"]}.'
                    f' Buffer played steps: {info["num_played_steps"]}.'
                    f' Env steps: {info["env_step"]}.'
                    f' Loss: {info["total_loss"]:.2f}',
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


if __name__ == "__main__":
    # gather arguments
    # example: python muzero.py --env cartpole --seed 666 --num_simulations 400 --training_steps 100000
    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_simulations', type=int, default=50)
    parser.add_argument('--training_steps', type=int, default=100000)
    parser.add_argument('--num_gpus', type=int, default=1, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=20, help='cpus available')
    parser.add_argument('--object_store_memory', type=int, default=20 * 1024 * 1024 * 1024,
                        help='object store memory')
    args = parser.parse_args()

    # init muzero
    config = {'seed': args.seed,
              'num_simulations': args.num_simulations,
              'training_steps': args.training_steps,
              'self.results_path': os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                                os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                      "%Y-%m-%d-%H-%M-%S") + '-' + str(args.seed) + '-' + str(args.num_simulations) +
                                                '-' + str(args.training_steps))
              }
    muzero = MuZero(args.env, config, args)
    muzero.train()

ray.shutdown()
