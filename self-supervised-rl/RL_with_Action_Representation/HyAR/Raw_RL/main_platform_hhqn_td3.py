import numpy as np
import torch
import gym
import argparse
import os
import gym_platform
from Raw_RL import utils
from agents import TD3
from agents import P_TD3
from agents import OurDDPG
from agents import P_DDPG
from agents import hhqn
from agents import hhqn_td3
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
import matplotlib.pyplot as plt


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate(env, policy, action_parameter_sizes, episodes=100):
    returns = []
    epioside_steps = []

    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            all_discrete_action, all_parameter_action = policy.select_action(state)
            discrete_action = np.argmax(all_discrete_action)
            offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
            parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
            action = pad_action(discrete_action, parameter_action)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        returns.append(total_reward)
    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean()


def run(args):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if args.env == "Platform-v0":
        print("env", args.env)
        env = gym.make(args.env)
        env = ScaledStateWrapper(env)
        env = PlatformFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.spaces[0].shape[0]

    discrete_action_dim = env.action_space.spaces[0].n
    action_parameter_sizes = np.array(
        [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = int(action_parameter_sizes.sum())
    discrete_emb_dim = discrete_action_dim
    parameter_emb_dim = parameter_action_dim
    max_action = 1.0
    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)

    kwargs = {
        "state_dim": state_dim,
        "discrete_action_dim": discrete_action_dim,
        "parameter_action_dim": parameter_action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "P-TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = P_TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "hhqn":
        policy = hhqn_td3.hhqn(**kwargs)
        print("hhqn")
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                       parameter_action_dim=1,
                                       all_parameter_action_dim=parameter_action_dim,
                                       discrete_emb_dim=discrete_emb_dim,
                                       parameter_emb_dim=parameter_emb_dim,
                                       max_size=int(1e5))

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_reward = 0.
    returns = []
    Reward = []
    Reward_100 = []
    max_steps = 250
    cur_step = 0
    Test_Reward_100 = []
    Test_epioside_step_100 = []
    Test_success_rate_100 = []
    total_timesteps = 0
    t = 0
    while total_timesteps < args.max_timesteps:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        all_discrete_action, all_parameter_action = policy.select_action(state)
        all_discrete_action = (
                all_discrete_action + np.random.normal(0, max_action * args.expl_noise, size=discrete_action_dim)
        ).clip(-max_action, max_action)
        all_parameter_action = (
                all_parameter_action + np.random.normal(0, max_action * args.expl_noise, size=parameter_action_dim)
        ).clip(-max_action, max_action)
        discrete_action = np.argmax(all_discrete_action)
        offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
        parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]

        action = pad_action(discrete_action, parameter_action)
        episode_reward = 0.
        for i in range(max_steps):
            cur_step = cur_step + 1
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            replay_buffer.add(state, discrete_action=None, parameter_action=None, all_parameter_action=None,
                              discrete_emb=all_discrete_action,
                              parameter_emb=all_parameter_action,
                              next_state=next_state,
                              state_next_state=None,
                              reward=reward, done=terminal)

            next_all_discrete_action, next_all_parameter_action = policy.select_action(next_state)

            next_all_discrete_action = (
                    next_all_discrete_action + np.random.normal(0, max_action * args.expl_noise,
                                                                size=discrete_action_dim)
            ).clip(-max_action, max_action)
            next_all_parameter_action = (
                    next_all_parameter_action + np.random.normal(0, max_action * args.expl_noise,
                                                                 size=parameter_action_dim)
            ).clip(-max_action, max_action)
            next_discrete_action = np.argmax(next_all_discrete_action)
            offset = np.array([action_parameter_sizes[i] for i in range(next_discrete_action)], dtype=int).sum()
            next_parameter_action = next_all_parameter_action[
                                    offset:offset + action_parameter_sizes[next_discrete_action]]

            next_action = pad_action(next_discrete_action, next_parameter_action)

            all_discrete_action, all_parameter_action, action = next_all_discrete_action, next_all_parameter_action, next_action
            state = next_state
            if cur_step >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                           np.array(returns[-100:]).mean()))

                while not terminal:
                    state = np.array(state, dtype=np.float32, copy=False)
                    all_discrete_action, all_parameter_action = policy.select_action(state)
                    discrete_action = np.argmax(all_discrete_action)
                    offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
                    parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
                    action = pad_action(discrete_action, parameter_action)
                    (state, _), reward, terminal, _ = env.step(action)

                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())
                Test_Reward, Test_epioside_step = evaluate(env, policy, action_parameter_sizes,
                                                           episodes=100)
                Test_Reward_100.append(Test_Reward)
                Test_epioside_step_100.append(Test_epioside_step)
            if terminal:
                break

        returns.append(episode_reward)
        total_reward += episode_reward

    print("save txt")
    dir = "result/hhqn/platform"
    data = "0817"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_hhqn_platform_"
    title2 = "Reward_100_hhqn_td3_platform_"
    title3 = "Test_Reward_100_hhqn_td3_platform_"
    title4 = "Test_epioside_step_100_hhqn_td3_platform_"
    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step_100,
               delimiter=',')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="hhqn")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='Platform-v0')  # platform goal HFO
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=128, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_episodes", default=50000, type=int)  # Max time steps to run environment
    parser.add_argument("--max_embedding_episodes", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for

    parser.add_argument("--epsilon_steps", default=1000, type=int)  # Max time steps to epsilon environment
    parser.add_argument("--expl_noise_initial", default=1.0)  # Std of Gaussian exploration noise 1.0
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise 0.1

    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
