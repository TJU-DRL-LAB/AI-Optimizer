import os
from pathlib import Path
import torch
from tqdm import trange

from pex.algorithms.iql import IQL
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, DEFAULT_DEVICE, sample_batch,
    eval_policy, set_default_device, get_env_and_dataset)


def main(args):
    torch.set_num_threads(1)
    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)

    env, dataset, _ = get_env_and_dataset(args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    if args.seed is not None:
        set_seed(args.seed, env=env)

    if torch.cuda.is_available():
        set_default_device()

    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)

    iql = IQL(
        critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        policy=policy,
        optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        target_update_rate=args.target_update_rate,
        discount=args.discount
    )

    for step in trange(args.num_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step + 1) % args.eval_period == 0:
            eval_policy(env, args.env_name, iql, args.max_episode_steps, args.eval_episode_num, args.seed)

    torch.save(iql.state_dict(), args.log_dir + '/offline_ckpt')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of training steps (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=10.0,
                        help='IQL inverse temperature')
    parser.add_argument('--eval_period', type=int, default=1000)
    parser.add_argument('--eval_episode_num', type=int, default=100,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    main(parser.parse_args())
