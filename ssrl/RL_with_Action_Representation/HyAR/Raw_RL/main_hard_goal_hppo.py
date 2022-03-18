import os
import click
import time
import numpy as np
import gym
import gym_goal
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
from common.wrappers import ScaledStateWrapper
from agents.pdqn import PDQNAgent
import matplotlib.pyplot as plt
import argparse
from agents.hppo_noshare import PPO
from agents.utils.ppo_utils import ReplayBufferPPO
import torch

import copy


def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2
    offset = c_rate[0] - 1 * median
    return median, offset


def true_action(act, act_param, c_rate):
    parameter_action_ = copy.deepcopy(act_param)
    median, offset = count_boundary(c_rate[act])
    parameter_action_ = parameter_action_ * median + offset
    return parameter_action_


# (act, params) (0, [array([-0.8359964 , -0.30679134], dtype=float32), array([0.]), array([0.])])
def pad_action(act, act_param):
    c_rate = [[-1.0, -0.6], [-0.6, -0.2], [0.2, 0.2], [0.2, 0.6], [0.6, 1.0]]
    # print("c_rate",c_rate[0])

    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    if act == 0:
        params[0][0] = act_param[0]
        params[0][1] = act_param[1]
    elif act == 1:
        act_param = true_action(0, act_param[2], c_rate)
        act_param = np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 2:
        act_param = true_action(1, act_param[3], c_rate)
        act_param = np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 3:
        act_param = true_action(2, act_param[4], c_rate)
        act_param = np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 4:
        act_param = true_action(3, act_param[5], c_rate)
        act_param = np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 5:
        act_param = true_action(4, act_param[6], c_rate)
        act_param = np.array([act_param])
        params[1] = act_param
        act = 1

    elif act == 6:
        act_param = true_action(0, act_param[7], c_rate)
        act_param = np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 7:
        act_param = true_action(1, act_param[8], c_rate)
        act_param = np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 8:
        act_param = true_action(2, act_param[9], c_rate)
        act_param = np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 9:
        act_param = true_action(3, act_param[10], c_rate)
        act_param = np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 10:
        act_param = true_action(4, act_param[11], c_rate)
        act_param = np.array([act_param])
        params[2] = act_param
        act = 2
    return (act, params)


def evaluate(env, policy, episodes=1000):
    returns = []
    epioside_steps = []
    possibility = []

    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            prob_discrete_action, discrete_action, parameter_action = policy.select_action(
                state, is_test=True)
            # offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
            # parameter_action_one = parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
            parameter_action_one = parameter_action
            action = pad_action(discrete_action, parameter_action_one)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        returns.append(total_reward)
        possibility.append((np.array(returns) == 50.).sum() / len(returns))

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}possibility: {np.array(possibility[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean(), np.array(possibility[-100:]).mean()


def run(args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("seed", args.seed)
    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)
    kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                               [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
    shoot_goal_left_weights = np.array([0.857346647646219686, 0])
    shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
    initial_weights = np.zeros((4, 17))
    initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
    initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
    initial_weights[2, 16] = shoot_goal_left_weights[1]
    initial_weights[3, 16] = shoot_goal_right_weights[1]

    initial_bias = np.zeros((4,))
    initial_bias[0] = kickto_weights[0, 0]
    initial_bias[1] = kickto_weights[1, 0]
    initial_bias[2] = shoot_goal_left_weights[0]
    initial_bias[3] = shoot_goal_right_weights[0]
    env = GoalFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    reward_scale = 1. / 50.

    dir = os.path.join("")
    env = Monitor(env, directory=os.path.join(dir, str(args.seed)), video_callable=False, write_upon_reset=False,
                  force=True)
    env.seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.spaces[0].shape[0]
    discrete_action_dim = 11
    # action_parameter_sizes = np.array(
    #     [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = 12
    max_action = 1.0

    if args.policy_name == "PPO":
        policy = PPO(state_dim, discrete_action_dim, parameter_action_dim, max_action, device)

    max_steps = 150
    total_reward = 0.
    returns = []
    start_time = time.time()
    replay_buffer = ReplayBufferPPO(obs_dim=state_dim, discrete_action_dim=1,
                                    parameter_action_dim=parameter_action_dim, size=args.epoch_steps)
    Reward = []
    Reward_100 = []
    epioside_steps = []
    epioside_steps_100 = []
    possibility = []
    total_timesteps = 0
    # for i in range(args.max_epiosides):
    while total_timesteps < args.max_timesteps:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        prob_discrete_action, discrete_action, parameter_action, raw_act, parameter_logp_t = policy.select_action(state)
        discrete_logp_t = np.max(prob_discrete_action)
        # offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
        # parameter_action_one = parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
        parameter_action_one = parameter_action
        v_t = policy.get_value(state)
        action = pad_action(discrete_action, parameter_action_one)
        episode_reward = 0.
        for j in range(max_steps):
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, done, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            replay_buffer.add(state, discrete_action, parameter_action, reward, v_t, discrete_logp_t, parameter_logp_t)

            next_prob_discrete_action, next_discrete_action, next_parameter_action, next_raw_act, next_parameter_logp_t = policy.select_action(
                next_state)
            next_discrete_logp_t = np.max(next_prob_discrete_action)
            # offset = np.array([action_parameter_sizes[i] for i in range(next_discrete_action)], dtype=int).sum()

            # next_parameter_action_one = next_parameter_action[
            #                             offset:offset + action_parameter_sizes[next_discrete_action]]
            next_parameter_action_one = next_parameter_action
            next_v_t = policy.get_value(next_state)

            next_action = pad_action(next_discrete_action, next_parameter_action_one)

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
                while not done:
                    state = np.array(state, dtype=np.float32, copy=False)
                    prob_discrete_action, discrete_action, parameter_action = policy.select_action(
                        state, is_test=True)
                    offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
                    parameter_action_one = parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
                    action = pad_action(discrete_action, parameter_action_one)
                    (state, _), reward, done, _ = env.step(action)

                Test_Reward, Test_epioside_step, Test_success = evaluate(env, policy, episodes=100)
                print(
                    '{0:5s} R:{1:.5f} P(S):{2:.4f} r100:{3:.4f} epioside_steps_100:{4:.4f}'.format(str(total_timesteps),
                                                                                                   total_reward / (
                                                                                                           total_timesteps + 1),
                                                                                                   Test_success,
                                                                                                   Test_Reward,
                                                                                                   Test_epioside_step))
                Reward_100.append(Test_Reward)
                Reward.append(total_reward / (total_timesteps + 1))
                possibility.append(Test_success)
                epioside_steps_100.append(Test_epioside_step)
            if done:
                break

        returns.append(episode_reward)
        total_reward += episode_reward

    dir = "result/HPPO/goal"
    data = "0829"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_hppo_goal_"
    title2 = "Reward_100_hppo_hard_goal_"
    title3 = "success_100_hppo_hard_goal_"
    title4 = "epioside_steps_100_hppo_hard_goal_"
    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), possibility, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), epioside_steps_100, delimiter=',')
    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, help='Random seed.', type=int)
    parser.add_argument("--policy_name", default="PPO")  # Policy name
    parser.add_argument("--env", default="InvertedDoublePendulum-v1")  # OpenAI gym environment name
    # parser.add_argument("--env_name", default="Walker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=500, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_epiosides", default=50000, type=float)  # Max time steps to run environment for
    parser.add_argument("--max_timesteps", default=300000, type=float)  # Max time steps to run environment for

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
