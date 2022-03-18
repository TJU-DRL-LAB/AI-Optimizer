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
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
import math


def pad_action(act, act_param):
    if act == 0:
        action = np.hstack(([5], act_param, [1], [0], [0], [0]))
    elif act == 1:
        action = np.hstack(([5], act_param, [0], [1], [0], [0]))
    elif act == 2:
        action = np.hstack(([5], act_param, [0], [0], [1], [0]))
    elif act == 3:
        action = np.hstack(([5], act_param, [0], [0], [0], [1]))
    return [action]


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate(env, policy, max_steps, episodes=100):
    returns = []
    success = []
    epioside_steps = []
    for _ in range(episodes):
        state = env.reset()
        t = 0
        total_reward = 0.
        flag = 0
        for j in range(max_steps):
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)[0]
            all_discrete_action, all_parameter_action = policy.select_action(state)
            discrete_action = np.argmax(all_discrete_action)
            action = pad_action(discrete_action, all_parameter_action)
            state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            total_reward += reward
            if reward > 4:
                flag = 1
                done=True
            if done or j == max_steps - 1:
                epioside_steps.append(j)
                break
        if flag == 1:
            success.append(1)
        else:
            success.append(0)
        returns.append(total_reward)

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-episodes:]).mean():.3f} {np.array(success[-episodes:]).mean():.3f} "
        f"{np.array(epioside_steps[-episodes:]).mean():.3f} ")
    print("---------------------------------------")
    return np.array(returns[-episodes:]).mean(), np.array(success[-episodes:]).mean(), np.array(
        epioside_steps[-episodes:]).mean()


def run(args):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = make_env(args.env)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    obs_n = env.reset()

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    print(obs_shape_n)
    torch.manual_seed(args.seed)

    state_dim = obs_shape_n[0][0]

    discrete_action_dim = 4
    # action_parameter_sizes = np.array(
    #     [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = 4
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
        policy = hhqn.hhqn(**kwargs)
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
    total_reward = 0.
    Reward = []
    Reward_100 = []
    Test_Reward = []
    max_steps = 30
    cur_step = 0
    flag = 0
    Test_success = []
    returns = []
    success = []
    Test_epioside_step = []
    total_timesteps = 0
    t = 0
    while total_timesteps < args.max_timesteps:
        state = obs_n
        state = np.array(state, dtype=np.float32, copy=False)[0]

        all_discrete_action, all_parameter_action = policy.select_action(state)
        all_discrete_action = (
                all_discrete_action + np.random.normal(0, max_action * args.expl_noise, size=discrete_action_dim)
        ).clip(-max_action, max_action)
        all_parameter_action = (
                all_parameter_action + np.random.normal(0, max_action * args.expl_noise, size=parameter_action_dim)
        ).clip(-max_action, max_action)
        discrete_action = np.argmax(all_discrete_action)

        action = pad_action(discrete_action, all_parameter_action)
        episode_reward = 0.
        flag = 0
        for i in range(max_steps):
            total_timesteps += 1
            cur_step = cur_step + 1
            next_state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            if reward > 4:
                flag = 1
                done=True
            next_state = np.array(next_state, dtype=np.float32, copy=False)[0]
            replay_buffer.add(state, discrete_action=None, parameter_action=None, all_parameter_action=None,
                              discrete_emb=all_discrete_action,
                              parameter_emb=all_parameter_action,
                              next_state=next_state,
                              state_next_state=None,
                              reward=reward, done=done)

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

            next_action = pad_action(next_discrete_action, next_all_parameter_action)

            all_discrete_action, all_parameter_action, action = next_all_discrete_action, next_all_parameter_action, next_action
            state = next_state
            if cur_step >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
            episode_reward += reward
            if total_timesteps % args.eval_freq == 0:
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f} success:{3:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                                           np.array(returns[-100:]).mean(),
                                                                           np.array(success[-100:]).mean()))
                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())
                Test_Reward_50, Test_success_rate, Test_epioside_step_50 = evaluate(env, policy,  max_steps=30, episodes=50)
                Test_Reward.append(Test_Reward_50)
                Test_success.append(Test_success_rate)
                Test_epioside_step.append(Test_epioside_step_50)


            if done or i == max_steps - 1:
                obs_n = env.reset()
                break
        if flag == 1:
            success.append(1)
        else:
            success.append(0)
        t += 1
        returns.append(episode_reward)
        total_reward += episode_reward

    print("save txt")
    dir = "result/hhqn/simple_move"
    data = "0715"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_hhqn_simple_move_"
    title2 = "Reward_100_hhqn_simple_move_"
    title3 = "Test_Reward_hhqn_simple_move_"
    title4 = "Test_success_hhqn_simple_move_"
    title5 = "Test_epioside_step_hhqn_simple_move_"

    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_success, delimiter=',')
    np.savetxt(os.path.join(redir, title5 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step, delimiter=',')


def make_env(scenario_name):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="hhqn")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='simple_move_4_direction')  # platform goal HFO
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=128, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_episodes", default=10000, type=int)  # Max time steps to run environment
    parser.add_argument("--max_embedding_episodes", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1000000, type=float)  # Max time steps to run environment for

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
    for i in range(1, 3):
        args.seed = i
        run(args)
