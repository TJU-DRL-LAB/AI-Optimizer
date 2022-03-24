import argparse
import numpy as np
import time
import pickle
import math
import os
import click
import gym
import gym_platform
from common import ClickPythonLiteralOption
import numpy as np
from agents.utils.noise import OrnsteinUhlenbeckActionNoise


def pad_action(act, act_param):
    if act == 0:
        action = np.hstack(([1], act_param * math.pi, [1], [0]))
    else:
        action = np.hstack(([1], act_param * math.pi, [0], [1]))

    return [action]


def evaluate(env, agent, max_steps, epsilon, episodes=1000):
    returns = []
    epioside_steps = []
    success = []
    agent.epsilon = 0.
    agent.noise = None
    for _ in range(episodes):
        state = env.reset()
        t = 0
        total_reward = 0.
        flag = 0
        for j in range(max_steps):
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)[0]
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            total_reward += reward
            if reward > 4:
                flag = 1
                done = True
            if reward == 0:
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
    agent.epsilon = epsilon
    agent.noise = OrnsteinUhlenbeckActionNoise(4, mu=0.,
                                               theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-episodes:]).mean():.3f} {np.array(success[-episodes:]).mean():.3f} "
        f"{np.array(epioside_steps[-episodes:]).mean():.3f} ")
    print("---------------------------------------")
    return np.array(returns[-episodes:]).mean(), np.array(success[-episodes:]).mean(), np.array(
        epioside_steps[-episodes:]).mean()


def run(args):
    env = make_env(args.scenario)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    obs_n = env.reset()

    if args.save_freq > 0 and args.save_dir:
        save_dir_ = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
        os.makedirs(save_dir_, exist_ok=True)

    assert not (args.save_frames and args.visualise)
    if args.visualise:
        assert args.render_freq > 0

    if args.save_frames:
        assert args.render_freq > 0
        vidir = os.path.join(args.save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    # env.seed(args.seed)
    np.random.seed(args.seed)

    print(obs_shape_n)

    from agents.pdqn_td3_MPE import PDQNAgent
    agent_class = PDQNAgent

    action_size = 2
    parameter_action_dim = 1
    agent = agent_class(
        obs_shape_n, action_size,
        parameter_action_dim=parameter_action_dim,
        batch_size=args.batch_size,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_actor_param=args.learning_rate_actor_param,
        epsilon_steps=args.epsilon_steps,
        gamma=args.gamma,
        tau_actor=args.tau_actor,
        tau_actor_param=args.tau_actor_param,
        clip_grad=args.clip_grad,
        indexed=args.indexed,
        weighted=args.weighted,
        average=args.average,
        random_weighted=args.random_weighted,
        initial_memory_threshold=args.initial_memory_threshold,
        use_ornstein_noise=args.use_ornstein_noise,
        replay_memory_size=args.replay_memory_size,
        epsilon_final=args.epsilon_final,
        inverting_gradients=args.inverting_gradients,
        zero_index_gradients=args.zero_index_gradients,
        seed=args.seed)

    print(agent)
    max_steps = 25
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    train_step = 0
    Reward = []
    Reward_100 = []
    success = []
    Test_success = []
    Test_epioside_step_100 = []
    total_timesteps = 0
    # for i in range(args.episodes):
    while total_timesteps < args.max_timesteps:

        state = obs_n
        state = np.array(state, dtype=np.float32, copy=False)[0]

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()
        flag = 0
        for j in range(max_steps):
            total_timesteps += 1
            train_step += 1
            next_state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            if reward > 4:
                flag = 1
                done = True
            if reward == 0:
                done = True

            next_state = np.array(next_state, dtype=np.float32, copy=False)[0]

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            # print("-----",next_act, next_act_param, next_all_action_parameters,next_action)

            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), done)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                Test_Reward_50, Test_success_rate, Test_epioside_step_50 = evaluate(env, agent,
                                                                                    max_steps=25,
                                                                                    epsilon=agent.epsilon,
                                                                                    episodes=50)
                print('{0:5s}  r100:{1:.4f} success:{2:.4f} Test_epioside_step:{3:.4f}'.format(str(total_timesteps),
                                                                                               Test_Reward_50,
                                                                                               Test_success_rate,
                                                                                               Test_epioside_step_50))
                Reward_100.append(Test_Reward_50)
                Test_success.append(Test_success_rate)
                Test_epioside_step_100.append(Test_epioside_step_50)

            if done or j == max_steps - 1:
                obs_n = env.reset()
                break

            # if visualise :
            #     time.sleep(0.1)
            #     env.render()
            #     continue

        agent.end_episode()

        if flag == 1:
            success.append(1)
        else:
            success.append(0)
        returns.append(episode_reward)
        total_reward += episode_reward


    dir = "result/PDQN/direction_catch"
    data = "0815"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_pdqn_direction_catch_"
    title2 = "Reward_100_pdqn_td3_direction_catch_"
    title3 = "success_pdqn_td3_direction_catch_"
    title4 = "Test_epioside_step_pdqn_td3_direction_catch_"

    # np.savetxt( os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")),  Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_success, delimiter=',')
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
    parser.add_argument('--scenario', default='simple_catch', help='name of the scenario script.', type=str)
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument("--eval_freq", default=2500, type=float)  # How often (time steps) we evaluate

    parser.add_argument('--episodes', default=10000, help='Number of epsiodes.', type=int)
    parser.add_argument("--max_timesteps", default=1000000, type=float)  # Max time steps to run environment for

    parser.add_argument('--batch-size', default=128, help='Minibatch size.', type=int)
    parser.add_argument('--gamma', default=0.95, help='Discount factor.', type=float)
    parser.add_argument('--inverting-gradients', default=True,
                        help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=128,
                        help='Number of transitions required to start learning.',
                        type=int)  # may have been running with 500??
    parser.add_argument('--use-ornstein-noise', default=True,
                        help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--replay-memory-size', default=100000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--epsilon-steps', default=1000,
                        help='Number of episodes over which to linearly anneal epsilon.',
                        type=int)
    parser.add_argument('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--tau-actor', default=0.01, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.',
                        type=float)
    parser.add_argument('--learning-rate-actor', default=0.001, help="discrete actor  learning rate.", type=float)
    parser.add_argument('--learning-rate-actor-param', default=0.0001, help="parameter actor  learning rate.",
                        type=float)
    # parser.add_argument('--scale-actions', default=True, help="Scale actions.", type=bool)
    # parser.add_argument('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
    parser.add_argument('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--indexed', default=False, help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False, help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False, help='Average weighted loss function.', type=bool)
    parser.add_argument('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--zero-index-gradients', default=False,
                        help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
                        type=bool)
    parser.add_argument('--save-freq', default=1000, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save-dir', default="results/model/mpe_direction_catch", help='Output directory.', type=str)
    parser.add_argument('--render-freq', default=1000, help='How often to render / save frames of an episode.',
                        type=int)
    parser.add_argument('--save-frames', default=False,
                        help="Save render frames from the environment. Incompatible with visualise.", type=bool)
    parser.add_argument('--visualise', default=True, help="Render game states. Incompatible with save-frames.",
                        type=bool)
    parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)

    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
