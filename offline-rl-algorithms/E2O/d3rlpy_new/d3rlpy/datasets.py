# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import List, Tuple
from urllib import request

import gym
import numpy as np

from d3rlpy.dataset import Episode, MDPDataset, Transition
from .envs import ChannelFirst

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long
PENDULUM_URL = f"{DROPBOX_URL}/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long


def get_cartpole(dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading cartpole.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(data_path)

    # environment
    env = gym.make("CartPole-v0")

    return dataset, env


def get_pendulum(dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading pendulum.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(data_path)

    # environment
    env = gym.make("Pendulum-v0")

    return dataset, env


def get_atari(env_name: str) -> Tuple[MDPDataset, gym.Env]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = ChannelFirst(gym.make(env_name))
        dataset = MDPDataset(discrete_action=True, **env.get_dataset())
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_atari_transitions(
    game_name: str, fraction: float = 0.01, index: int = 0
) -> Tuple[List[Transition], gym.Env]:
    """Returns atari dataset as a list of Transition objects and envrironment.

    The dataset is provided through d4rl-atari.
    The difference from ``get_atari`` function is that this function will
    sample transitions from all epochs.
    This function is necessary for reproducing Atari experiments.

    .. code-block:: python

        from d3rlpy.datasets import get_atari_transitions

        # get 1% of transitions from all epochs (1M x 50 epoch x 1% = 0.5M)
        dataset, env = get_atari_transitions('breakout', fraction=0.01)

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        game_name: Atari 2600 game name in lower_snake_case.
        fraction: fraction of sampled transitions.
        index: index to specify which trial to load.

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.

    """
    try:
        import d4rl_atari

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        transitions = []
        for i in range(50):
            env = gym.make(
                f"{game_name}-epoch-{i + 1}-v{index}", sticky_action=True
            )
            dataset = MDPDataset(discrete_action=True, **env.get_dataset())
            episodes = list(dataset.episodes)

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            copied_episodes = []
            for episode in episodes:
                copied_episode = Episode(
                    observation_shape=tuple(episode.get_observation_shape()),
                    action_size=episode.get_action_size(),
                    observations=episode.observations.copy(),
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminal=episode.terminal,
                )
                copied_episodes.append(copied_episode)

                num_data += len(copied_episode)
                if num_data > num_transitions_per_epoch:
                    break

            transitions_per_epoch = []
            for episode in copied_episodes:
                transitions_per_epoch += episode.transitions
            transitions += transitions_per_epoch[:num_transitions_per_epoch]

        return transitions, ChannelFirst(env)
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_d4rl(env_name: str, norm_state: bool = False, change_reward: bool = False) -> Tuple[MDPDataset, gym.Env]:
    """Returns d4rl dataset and envrironment.
    The dataset is provided through d4rl.
    .. code-block:: python
        from d3rlpy.datasets import get_d4rl
        dataset, env = get_d4rl('hopper-medium-v0')
    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl
    Args:
        env_name: environment id of d4rl dataset.
        norm_state
    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.
    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset()

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        timeouts = dataset["timeouts"]
        episode_terminals = np.logical_or(terminals, timeouts)

        if norm_state:
            # hopper-medium-v2
            obs_mean = np.array([1.3112686, -0.08469262, -0.53825426, -0.07201052,  0.04934485,
                                 2.106582 , -0.15014714,  0.00878511, -0.28485158, -0.18539158,
                                 -0.28468254], dtype=np.float32)
            obs_std = np.array([0.17790356, 0.05444648, 0.2129813, 0.14529777, 0.61242145,
                                0.8517834 , 1.4514914 , 0.675177 , 1.5362306 , 1.6160218 ,
                                5.6071143 ], dtype=np.float32)
            observations = (observations - obs_mean) / obs_std

        if change_reward:
            rewards -= 1

        mdp_dataset = MDPDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )

        return mdp_dataset, env

    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]


def get_dataset(env_name: str, norm_state: bool = False, change_reward: bool = False) -> Tuple[MDPDataset, gym.Env]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole-replay
    - cartpole-random
    - pendulum-replay
    - pendulum-random
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        norm_state

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if env_name == "cartpole-replay":
        return get_cartpole(dataset_type="replay")
    elif env_name == "cartpole-random":
        return get_cartpole(dataset_type="random")
    elif env_name == "pendulum-replay":
        return get_pendulum(dataset_type="replay")
    elif env_name == "pendulum-random":
        return get_pendulum(dataset_type="random")
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(env_name)
    elif re.match(r"hopper|halfcheetah|walker|ant|pen|door|maze", env_name):
        return get_d4rl(env_name, norm_state, change_reward)
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(env_name)
    raise ValueError(f"Unrecognized env_name: {env_name}.")
