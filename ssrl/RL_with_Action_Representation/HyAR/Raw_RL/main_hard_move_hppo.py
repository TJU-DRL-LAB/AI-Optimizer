import argparse
import numpy as np
import time
import pickle

import os
import click
import gym
import gym_platform
from common import ClickPythonLiteralOption
import numpy as np
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
import torch
from agents.hppo_noshare import PPO
from agents.utils.ppo_utils import ReplayBufferPPO


def pad_action(act, act_param, action_n_dim):
    action = np.hstack(([8], [act], [action_n_dim])).tolist()

    action.append(act_param)
    action = np.array(action)
    # print("action",action[3])

    return [action]


def evaluate(env, policy, max_steps,action_n_dim, episodes=1000):
    returns = []
    epioside_steps = []
    success = []

    for _ in range(episodes):
        state = env.reset()
        t = 0
        total_reward = 0.
        flag = 0
        for j in range(max_steps):
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)[0]
            prob_discrete_action, discrete_action, parameter_action = policy.select_action(
                state, is_test=True)
            action = pad_action(discrete_action, parameter_action, action_n_dim)
            state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            total_reward += reward
            if reward > 4:
                flag = 1
                done = True
            if done or j == max_steps - 1:
                epioside_steps.append(j)
                break
        if flag == 1:
            success.append(1)
        else:
            success.append(0)
        epioside_steps.append(t)
        returns.append(total_reward)

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-episodes:]).mean():.3f} {np.array(success[-episodes:]).mean():.3f} "
        f"{np.array(epioside_steps[-episodes:]).mean():.3f} ")
    print("---------------------------------------")
    return np.array(returns[-episodes:]).mean(), np.array(success[-episodes:]).mean(), np.array(
        epioside_steps[-episodes:]).mean()


def run(args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    obs_n = env.reset()
    print("seed", args.seed)

    # Set seeds
    # env.seed(args.seed)
    np.random.seed(args.seed)
    print(obs_shape_n)
    torch.manual_seed(args.seed)

    state_dim = obs_shape_n[0][0]
    action_n_dim = args.action_n_dim
    print("action_n_dim",action_n_dim)
    discrete_action_dim = 2 ** action_n_dim
    # action_para_size=4
    parameter_action_dim = 2 ** action_n_dim
    max_action = 1.0

    np.random.seed(args.seed)
    print(obs_shape_n)
    if args.policy_name == "PPO":
        policy = PPO(state_dim, discrete_action_dim, parameter_action_dim, max_action, device)
    replay_buffer = ReplayBufferPPO(obs_dim=state_dim, discrete_action_dim=1,
                                    parameter_action_dim=parameter_action_dim, size=args.epoch_steps)
    Reward = []
    Reward_100 = []
    max_steps = 30
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    total_timesteps = 0
    train_step = 0
    success = []
    possibilities = []
    Test_epioside_step = []
    Test_epioside_step_100 = []
    # for i in range(args.max_epiosides):
    while total_timesteps < args.max_timesteps:

        state = obs_n
        state = np.array(state, dtype=np.float32, copy=False)[0]
        prob_discrete_action, discrete_action, parameter_action, raw_act, parameter_logp_t = policy.select_action(state)
        discrete_logp_t = np.max(prob_discrete_action)
        v_t = policy.get_value(state)
        action = pad_action(discrete_action, parameter_action, action_n_dim)
        episode_reward = 0.
        flag = 0
        for j in range(max_steps):
            total_timesteps += 1
            next_state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            if reward > 4:
                flag = 1
                done = True
            next_state = np.array(next_state, dtype=np.float32, copy=False)[0]

            replay_buffer.add(state, discrete_action, parameter_action, reward, v_t, discrete_logp_t, parameter_logp_t)

            next_prob_discrete_action, next_discrete_action, next_parameter_action, next_raw_act, next_parameter_logp_t = policy.select_action(
                next_state)
            next_discrete_logp_t = np.max(next_prob_discrete_action)

            next_v_t = policy.get_value(next_state)

            next_action = pad_action(next_discrete_action, next_parameter_action, action_n_dim)

            discrete_action, parameter_action, v_t, discrete_logp_t, parameter_logp_t = next_discrete_action, next_parameter_action, next_v_t, next_discrete_logp_t, next_parameter_logp_t
            action = next_action
            state = next_state

            episode_reward += reward
            is_to_update = (total_timesteps % args.epoch_steps == 0)

            if done or is_to_update:
                # if trajectory didn't reach terminal state, bootstrap value target

                last_val = reward if done else policy.get_value(next_state)
                replay_buffer.finish_path(last_val)
                if is_to_update:
                    losses = policy.train(replay_buffer, c_epoch=10, a_epoch=2)
                    # print("discrete_action, parameter_action",discrete_action, parameter_action)
                    # print("losses", losses)
                    replay_buffer.reset()

            if total_timesteps % args.eval_freq == 0:
                Test_Reward_50, Test_success_rate, Test_epioside_step_50 = evaluate(env, policy,
                                                                                    action_n_dim=action_n_dim,
                                                                                    max_steps=30,
                                                                                    episodes=50)
                print('{0:5s}  r100:{1:.4f} success:{2:.4f} Test_epioside_step:{3:.4f}'.format(str(total_timesteps),
                                                                                               Test_Reward_50,
                                                                                               Test_success_rate,
                                                                                               Test_epioside_step_50))
                Reward_100.append(Test_Reward_50)
                possibilities.append(Test_success_rate)
                Test_epioside_step_100.append(Test_epioside_step_50)

            if done or j == max_steps - 1:
                obs_n = env.reset()
                Test_epioside_step.append(j)
                break

        # if flag == 1:
        #     success.append(1)
        # else:
        #     success.append(0)
        # returns.append(episode_reward)
        # total_reward += episode_reward

        # if i % 100 == 0:
        #     Test_Reward_50, Test_success_rate, Test_epioside_step_50 = evaluate(env, policy,
        #                                                                         max_steps=30,
        #                                                                         episodes=50)
        #     print('{0:5s} R:{1:.4f} r100:{2:.4f} success:{3:.4f} Test_epioside_step:{4:.4f}'.format(str(i), total_reward / (i + 1),
        #                                                                        Test_Reward_50,
        #                                                                        Test_success_rate,Test_epioside_step_50))
        #     Reward.append(total_reward / (i + 1))
        #     Reward_100.append(Test_Reward_50)
        #     possibilities.append(Test_success_rate)
        #     Test_epioside_step_100.append(Test_epioside_step_50)

    dir = "result/HPPO/simple_move_4_direction_v1"
    data = "0704"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_hppo_simple_move_4_direction_"
    title2 = "Reward_100_hppo_simple_move_4_direction_v1_noshare_"
    title3 = "Test_success_hppo_simple_move_4_direction_v1_noshare_"
    title4 = "Test_epioside_step_hppo_simple_move_4_direction_v1_noshare_"

    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), possibilities, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step_100,
               delimiter=',')

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="PPO")  # Policy name
    parser.add_argument('--seed', default=0, help='Random seed.', type=int)
    parser.add_argument('--action_n_dim', default=4, help='action_n_dim.', type=int)

    parser.add_argument("--env", default="simple_move_4_direction_v1")  # OpenAI gym environment name
    # parser.add_argument("--env_name", default="Walker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=2500, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_epiosides", default=50000, type=float)  # Max time steps to run environment for
    parser.add_argument("--max_timesteps", default=1000000, type=float)  # Max time steps to run environment for

    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--epoch-steps", default=50, type=int)  # num of steps to collect for each training iteration
    parser.add_argument("--is-state-norm", default=0, type=int)  # is use state normalization

    parser.add_argument("--gpu-no", default='-1', type=str)  # Frequency of delayed policy updates
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
