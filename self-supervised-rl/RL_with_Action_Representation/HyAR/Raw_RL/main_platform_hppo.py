import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
import argparse
import numpy as np
import torch
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from agents.hppo_noshare import PPO
from agents.utils.ppo_utils import ReplayBufferPPO


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, policy,action_parameter_sizes, episodes=1000):
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
            prob_discrete_action, discrete_action, parameter_action = policy.select_action(
                state,is_test=True)
            offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
            parameter_action_one = parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
            action = pad_action(discrete_action, parameter_action_one)
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("seed",args.seed)

    env = gym.make('Platform-v0')
    initial_params_ = [3., 10., 400.]
    for a in range(env.action_space.spaces[0].n):
        initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)
    dir = ""
    env = Monitor(env, directory=os.path.join(dir, str(args.seed)), video_callable=False, write_upon_reset=False,
                  force=True)
    env.seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.spaces[0].shape[0]
    discrete_action_dim = env.action_space.spaces[0].n
    action_parameter_sizes = np.array(
        [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = int(action_parameter_sizes.sum())
    max_action = 1.0
    print(env.observation_space)
    if args.policy_name == "PPO":
        policy = PPO(state_dim, discrete_action_dim, parameter_action_dim, max_action, device)

    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()

    replay_buffer = ReplayBufferPPO(obs_dim=state_dim, discrete_action_dim=1,
                                              parameter_action_dim=parameter_action_dim, size=args.epoch_steps)
    Return = []
    last_100_return = []
    total_timesteps = 0
    Reward_100 = []
    Reward = []
    epioside_steps = []
    epioside_steps_100 = []
    # for i in range(args.max_epiosides):
    while total_timesteps < args.max_timesteps:

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        prob_discrete_action,discrete_action, parameter_action, raw_act, parameter_logp_t = policy.select_action(state)
        discrete_logp_t = np.max(prob_discrete_action)
        offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
        parameter_action_one = parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
        v_t = policy.get_value(state)
        action = pad_action(discrete_action, parameter_action_one)
        episode_reward = 0.
        for j in range(max_steps):
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, done, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            replay_buffer.add(state, discrete_action, parameter_action, reward, v_t, discrete_logp_t, parameter_logp_t)

            next_prob_discrete_action,next_discrete_action, next_parameter_action, next_raw_act, next_parameter_logp_t = policy.select_action(
                next_state)
            next_discrete_logp_t = np.max(next_prob_discrete_action)
            offset = np.array([action_parameter_sizes[i] for i in range(next_discrete_action)], dtype=int).sum()

            next_parameter_action_one = next_parameter_action[offset:offset + action_parameter_sizes[next_discrete_action]]

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
                    losses = policy.train(replay_buffer, c_epoch=10, a_epoch=2) # a_epoch:1-5
                    # print("discrete_action, parameter_action",discrete_action, parameter_action)
                    # print("losses",losses)
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

                Test_Reward, Test_epioside_step = evaluate(env, policy, action_parameter_sizes, episodes=100)

                print('{0:5s} R:{1:.4f} r100:{2:.4f} steps:{3:.4f}'.format(str(total_timesteps), total_reward / (total_timesteps + 1), Test_Reward,
                                                                           Test_epioside_step))
                Reward_100.append(Test_Reward)
                Reward.append(total_reward / (total_timesteps + 1))
                epioside_steps_100.append(Test_epioside_step)
            if done:
                break


        returns.append(episode_reward)
        total_reward += episode_reward
        # if i % 100 == 0:
        #     Test_Reward, Test_epioside_step = evaluate(env, policy,action_parameter_sizes, episodes=100)
        #
        #     print('{0:5s} R:{1:.4f} r100:{2:.4f} steps:{3:.4f}'.format(str(i), total_reward / (i + 1), Test_Reward,
        #                                                                Test_epioside_step))
        #     Reward_100.append(Test_Reward)
        #     Reward.append(total_reward / (i + 1))
        #     epioside_steps_100.append(Test_epioside_step)

    dir = "result/HPPO/platform"
    data = "0703"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_hppo_platform_"
    title2 = "Reward_100_hppo_platform_"
    title3 = "epioside_steps_100_hppo_platform_"

    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), epioside_steps_100, delimiter=',')

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="PPO")  # Policy name
    parser.add_argument("--env", default="InvertedDoublePendulum-v1")  # OpenAI gym environment name
    # parser.add_argument("--env_name", default="Walker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=500, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_epiosides", default=40000, type=float)  # Max time steps to run environment for
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for

    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--epoch-steps", default=50, type=int)  # num of steps to collect for each training iteration
    parser.add_argument("--is-state-norm", default=0, type=int)  # is use state normalization

    parser.add_argument("--gpu-no", default='-1', type=str)  # Frequency of delayed policy updates
    args = parser.parse_args()
    for i in range(0, 5):
        args.seed = i
        run(args)
