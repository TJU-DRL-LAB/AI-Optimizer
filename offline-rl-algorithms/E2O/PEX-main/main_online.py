from pathlib import Path

import gym
import d4rl
import numpy as np
import itertools
import os
import torch
from tqdm import trange

from pex.algorithms.pex import PEX
from pex.algorithms.iql_online import IQL_online
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, ReplayMemory, torchify, eval_policy, torchify, DEFAULT_DEVICE,
    get_batch_from_dataset_and_buffer,
    eval_policy, set_default_device, get_env_and_dataset)


def main(args):
    torch.set_num_threads(1)

    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)

    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    dataset_size = dataset['observations'].shape[0]
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    if args.seed is not None:
        set_seed(args.seed, env=env)

    if torch.cuda.is_available():
        set_default_device()

    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)

    algorithm_option = args.algorithm.upper()

    if algorithm_option == "SCRATCH":
        double_buffer = False
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "BUFFER":
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "DIRECT":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path
        )

    elif algorithm_option == "PEX":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEX(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )

    memory = ReplayMemory(args.replay_size, args.seed)

    total_numsteps = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            action = alg.select_action(torchify(state).to(DEFAULT_DEVICE)).detach().cpu().numpy()
            if len(memory) > args.initial_collection_steps:
                for i in range(args.updates_per_step):
                    alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            reward_for_replay = reward_transformer(reward)

            terminal = 0 if episode_steps == env._max_episode_steps else float(done)
            memory.push(state, action, reward_for_replay, next_state, terminal)
            state = next_state

            if total_numsteps % args.eval_period == 0 and args.eval is True:

                print("Episode: {}, total env-steps: {}".format(i_episode, total_numsteps))
                eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num, args.seed)

        if total_numsteps > args.total_env_steps:
            break

        env.close()

    # torch.save(alg.state_dict(), args.log_dir + '/{}_online_ckpt'.format(args.algorithm))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex']
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=10.0,
                        help='IQL inverse temperature')
    parser.add_argument('--ckpt_path', default=None,
                        help='path to the offline checkpoint')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--total_env_steps', type=int, default=250001, metavar='N',
                        help='total number of env steps (default: 1000000)')
    parser.add_argument('--initial_collection_steps', type=int, default=5000, metavar='N',
                        help='Initial environmental steps before training starts (default: 5000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--inv_temperature', type=float, default=10, metavar='G',
                        help='inverse temperature for PEX action selection (default: 10)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--eval_period', type=int, default=1000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)

    main(parser.parse_args())
