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
from agents.pdqn_hard_goal import PDQNAgent
import matplotlib.pyplot as plt
import argparse
from agents.utils.noise import OrnsteinUhlenbeckActionNoise

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
    c_rate = [[-1.0, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0], [0, 0.2], [0.2, 0.4], [0.4, 0.6],
              [0.6, 0.8], [0.8, 1.0]]
    c_rate = [[-1.0, -0.6], [-0.6, -0.2], [0.2, 0.2], [0.2, 0.6], [0.6, 1.0]]
    # print("c_rate",c_rate[0])

    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    if act == 0:
        params[0][0] = act_param[0]
        params[0][1] = act_param[1]
    elif act == 1:
        act_param=true_action(0, act_param[2], c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 2:
        act_param=true_action(1, act_param[3], c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 3:
        act_param=true_action(2, act_param[4], c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 4:
        act_param=true_action(3, act_param[5], c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 5:
        act_param=true_action(4, act_param[6], c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1

    elif act == 6:
        act_param=true_action(0, act_param[7], c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 7:
        act_param=true_action(1, act_param[8], c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 8:
        act_param=true_action(2, act_param[9], c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 9:
        act_param=true_action(3, act_param[10], c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 10:
        act_param=true_action(4, act_param[11], c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    return (act, params)


def evaluate(env, agent, epsilon, episodes=1000):
    returns = []
    epioside_steps = []
    possibility = []
    agent.epsilon = 0.
    agent.noise = None
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        returns.append(total_reward)
        possibility.append((np.array(returns) == 50.).sum() / len(returns))
    agent.epsilon = epsilon
    agent.noise = OrnsteinUhlenbeckActionNoise(12, mu=0.,
                                               theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}possibility: {np.array(possibility[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean(), np.array(possibility[-100:]).mean()


def run(args):
    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)

    if args.save_freq > 0 and args.save_dir:
        save_dir_ = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
        print("save_dir", save_dir_)
        os.makedirs(save_dir_, exist_ok=True)
    assert not (args.save_frames and args.visualise)
    if args.visualise:
        assert args.render_freq > 0
    if args.save_frames:
        assert args.render_freq > 0
        vidir = os.path.join(args.save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    if args.scale_actions:
        kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                   [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        shoot_goal_left_weights = np.array([0.857346647646219686, 0])
        shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
    else:
        xfear = 50.0 / PITCH_LENGTH
        yfear = 50.0 / PITCH_WIDTH
        caution = 5.0 / PITCH_WIDTH
        kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
        shoot_goal_left_weights = np.array([GOAL_WIDTH / 2 - 1, 0])
        shoot_goal_right_weights = np.array([-GOAL_WIDTH / 2 + 1, 0])

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

    if not args.scale_actions:
        # rescale initial action-parameters for a scaled state space
        for a in range(env.action_space.spaces[0].n):
            mid = (env.observation_space.spaces[0].high + env.observation_space.spaces[0].low) / 2.
            initial_bias[a] += np.sum(initial_weights[a] * mid)
            initial_weights[a] = initial_weights[a] * env.observation_space.spaces[0].high - initial_weights[
                a] * mid

    env = GoalFlattenedActionWrapper(env)
    if args.scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    dir = os.path.join(args.save_dir, args.title)
    env = Monitor(env, directory=os.path.join(dir, str(args.seed)), video_callable=False, write_upon_reset=False,
                  force=True)

    env.seed(args.seed)
    np.random.seed(args.seed)
    # print("seed",np.random.seed(seed))
    agent_class = PDQNAgent

    action_size = 11
    parameter_action_dim = 12
    agent = agent_class(
        observation_space=env.observation_space.spaces[0], action_space=action_size,
        parameter_action_dim=parameter_action_dim,
        batch_size=args.batch_size,
        learning_rate_actor=args.learning_rate_actor,  # 0.0001
        learning_rate_actor_param=args.learning_rate_actor_param,  # 0.001
        epsilon_steps=args.epsilon_steps,
        epsilon_final=args.epsilon_final,
        gamma=args.gamma,
        clip_grad=args.clip_grad,
        indexed=args.indexed,
        average=args.average,
        random_weighted=args.random_weighted,
        tau_actor=args.tau_actor,
        weighted=args.weighted,
        tau_actor_param=args.tau_actor_param,
        initial_memory_threshold=args.initial_memory_threshold,
        use_ornstein_noise=args.use_ornstein_noise,
        replay_memory_size=args.replay_memory_size,
        inverting_gradients=args.inverting_gradients,
        zero_index_gradients=args.zero_index_gradients,
        seed=args.seed)

    # if args.initialise_params:
    # agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    # print("initial_bias",initial_bias)
    print(agent)
    max_steps = 150
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    Reward = []
    Reward_100 = []
    epioside_steps = []
    epioside_steps_100 = []
    possibility = []
    total_timesteps = 0
    # for i in range(episodes):
    while total_timesteps < args.max_timesteps:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent.act(state)
        # print('sdfsdfs')
        # print(act, act_param)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            # r = reward * reward_scale
            if total_timesteps%100==0:
                print("next_action",next_action)
            r = reward
            agent.step(state, (act, all_action_parameters), r, next_state,
                       (next_act, next_all_action_parameters), terminal)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                while not terminal:  # 如果每结束需要继续推演
                    state = np.array(state, dtype=np.float32, copy=False)
                    act, act_param, all_action_parameters = agent.act(state)
                    action = pad_action(act, act_param)
                    (state, _), reward, terminal, _ = env.step(action)

                Test_Reward, Test_epioside_step, Test_success = evaluate(env, agent, agent.epsilon, episodes=100)
                print('{0:5s}  P(S):{1:.4f} r100:{2:.4f} epioside_steps_100:{3:.4f}'.format(str(total_timesteps),
                                                                                            Test_success,
                                                                                            Test_Reward,
                                                                                            Test_epioside_step))
                Reward_100.append(Test_Reward)
                # Reward.append(total_reward / (i + 1))
                possibility.append(Test_success)
                epioside_steps_100.append(Test_epioside_step)

            if terminal:
                break
        agent.end_episode()

        returns.append(episode_reward)
        total_reward += episode_reward

    dir = "result/PDQN/goal"
    data = "0829"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)

    # title1 = "Reward_pdqn_hard_goal_"
    title2 = "Reward_100_pdqn_hard_goal_"
    title3 = "success_100_pdqn_hard_goal_"
    title4 = "epioside_steps_100_pdqn_hard_goal_"
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
    parser.add_argument("--eval_freq", default=500, type=float)  # How often (time steps) we evaluate

    parser.add_argument("--max_timesteps", default=300000, type=float)  # Max time steps to run environment for
    parser.add_argument('--batch-size', default=128, help='Minibatch size.', type=int)
    parser.add_argument('--gamma', default=0.95, help='Discount factor.', type=float)
    parser.add_argument('--inverting-gradients', default=True,
                        help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=128,
                        help='Number of transitions required to start learning.',
                        type=int)
    parser.add_argument('--use-ornstein-noise', default=True,
                        help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--replay-memory-size', default=20000, help='Replay memory transition capacity.', type=int)
    parser.add_argument('--epsilon-steps', default=1000,
                        help='Number of episodes over which to linearly anneal epsilon.',
                        type=int)
    parser.add_argument('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.',
                        type=float)
    parser.add_argument('--learning-rate-actor', default=0.001, help="discrete actor  learning rate.", type=float)
    parser.add_argument('--learning-rate-actor-param', default=0.00001, help="parameter actor  learning rate.",
                        type=float)
    parser.add_argument('--scale-actions', default=True, help="Scale actions.", type=bool)
    parser.add_argument('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
    parser.add_argument('--reward-scale', default=1. / 50., help="Reward scaling factor.", type=float)
    parser.add_argument('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--multipass', default=False,
                        help='Separate action-parameter inputs using multiple Q-network passes.',
                        type=bool)
    parser.add_argument('--indexed', default=False, help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False, help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False, help='Average weighted loss function.', type=bool)
    parser.add_argument('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--split', default=False, help='Separate action-parameter inputs.', type=bool)
    parser.add_argument('--zero-index-gradients', default=False,
                        help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
                        type=bool)
    parser.add_argument('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
    parser.add_argument('--save-freq', default=2000, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save-dir', default="results/model/goal", help='Output directory.', type=str)
    parser.add_argument('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
    parser.add_argument('--save-frames', default=False,
                        help="Save render frames from the environment. Incompatible with visualise.", type=bool)
    parser.add_argument('--visualise', default=False, help="Render game states. Incompatible with save-frames.",
                        type=bool)
    parser.add_argument('--title', default="PDQN", help="Prefix of output files", type=str)
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
