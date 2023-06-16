import random
import time

import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from loguru import logger

import utils
import ISPI


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    logger.info("---------------------------------------")
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    logger.info("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="ISPI")  # Policy name
    parser.add_argument("--env", default="hopper-medium-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--eval_episodes", default=10, type=int)  # eval episodes, Mujoco 10, Antmaze 100

    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=5e5, type=int)  # How often (time steps) we save model
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--critic_lr", default=3e-4, type=float)  # critic learning rate
    parser.add_argument("--actor_lr", default=3e-4, type=float)  # actor learning rate
    # DTD3 + BC
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--aweight", default=0.5, type=float, help='The weighting coefficient')
    parser.add_argument("--normalize", action="store_true", help='state normalization')
    parser.add_argument("--reward_scale", default=1., type=float, help='reward scale')
    parser.add_argument("--reward_bias", default=0., type=float, help='reward bias')
    parser.add_argument("--reward_standardize", action="store_true", help='reward standardize')

    args = parser.parse_args()

    if 'antmaze' in args.env:
        args.eval_episodes = 100
        args.eval_freq = 5e4

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    logger.info("---------------------------------------")
    logger.info(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    logger.info("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists(f"./models/{args.env}"):
        os.makedirs(f"./models/{args.env}")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device": device,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "critic_lr": args.critic_lr,
        "actor_lr": args.actor_lr,
        # DTD3 + BC
        "alpha": args.alpha,
        "a_weight": args.aweight,
    }

    # Initialize policy
    policy = ISPI.ISPI(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), reward_scale=args.reward_scale, reward_bias=args.reward_bias, standardize=args.reward_standardize)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    # evaluations = []
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            logger.info(f"Time steps: {t + 1}")
            performance_return = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
        if args.save_model and (t == 0 or (t + 1) % args.save_freq == 0):
            save_file_name = f"{file_name}_{t + 1}" if t != 0 else f"{file_name}_0"
            policy.save(f"./models/{save_file_name}")
