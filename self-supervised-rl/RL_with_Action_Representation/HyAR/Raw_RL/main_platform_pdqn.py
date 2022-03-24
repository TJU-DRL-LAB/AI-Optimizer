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

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from agents.utils.noise import OrnsteinUhlenbeckActionNoise


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, epsilon, episodes=1000):
    returns = []
    epioside_steps = []
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

    agent.epsilon = epsilon
    # print("agent.epsilon",agent.epsilon)
    agent.noise = OrnsteinUhlenbeckActionNoise(3, mu=0.,
                                               theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)

    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean()


def run(args):
    if args.save_freq > 0 and args.save_dir:
        save_dir = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (args.save_frames and args.visualise)
    if args.visualise:
        assert args.render_freq > 0
    if args.save_frames:
        assert args.render_freq > 0
        vidir = os.path.join(save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    env = gym.make('Platform-v0')
    initial_params_ = [3., 10., 400.]
    if args.scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                    env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.

    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if args.scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir, args.title)
    env = Monitor(env, directory=os.path.join(dir, str(args.seed)), video_callable=False, write_upon_reset=False,
                  force=True)
    env.seed(args.seed)
    np.random.seed(args.seed)

    print(env.observation_space)

    from agents.pdqn import PDQNAgent
    agent_class = PDQNAgent

    agent = agent_class(
        env.observation_space.spaces[0], env.action_space,
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

    if args.initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0

    Reward_100 = []
    Reward = []
    epioside_steps = []
    epioside_steps_100 = []
    total_timesteps = 0
    # for i in range(episodes):
    while total_timesteps < args.max_timesteps:
        state, _ = env.reset()

        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_action_parameters = agent.act(state)
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
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            if total_timesteps % args.eval_freq == 0 :
                while not terminal: #如果每结束需要继续推演
                    state = np.array(state, dtype=np.float32, copy=False)
                    act, act_param, all_action_parameters = agent.act(state)
                    action = pad_action(act, act_param)
                    (state, _), reward, terminal, _ = env.step(action)

                Test_Reward, Test_epioside_step = evaluate(env, agent, agent.epsilon, episodes=100)

                print('{0:5s}  r100:{1:.4f} steps:{2:.4f}'.format(str(total_timesteps), Test_Reward,
                                                                  Test_epioside_step))
                Reward_100.append(Test_Reward)
                Reward.append(total_reward / (i + 1))
                epioside_steps_100.append(Test_epioside_step)

            if terminal:
                break
        agent.end_episode()

        returns.append(episode_reward)
        total_reward += episode_reward

        # if i % 100 == 0:
        #     Test_Reward, Test_epioside_step = evaluate(env, agent,agent.epsilon, episodes=100)
        #
        #     print('{0:5s} R:{1:.4f} r100:{2:.4f} steps:{3:.4f}'.format(str(i), total_reward / (i + 1), Test_Reward, Test_epioside_step))
        #     Reward_100.append(Test_Reward)
        #     Reward.append(total_reward / (i + 1))
        #     epioside_steps_100.append(Test_epioside_step)

    dir = "result/PDQN/platform"
    data = "0703"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    title1 = "Reward_pdqn_platform_"
    title2 = "Reward_100_pdqn_platform_"
    title3 = "epioside_steps_100_pdqn_platform_"

    np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), epioside_steps_100, delimiter=',')

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument("--eval_freq", default=500, type=float)  # How often (time steps) we evaluate
    parser.add_argument('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.',
                        type=int)
    parser.add_argument('--episodes', default=50000, help='Number of epsiodes.', type=int)
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for

    parser.add_argument('--batch-size', default=128, help='Minibatch size.', type=int)
    parser.add_argument('--gamma', default=0.9, help='Discount factor.', type=float)
    parser.add_argument('--inverting-gradients', default=True,
                        help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=500,
                        help='Number of transitions required to start learning.',
                        type=int)  # may have been running with 500??
    parser.add_argument('--use-ornstein-noise', default=True,
                        help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--epsilon-steps', default=1000,
                        help='Number of episodes over which to linearly anneal epsilon.',
                        type=int)
    parser.add_argument('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--tau-actor', default=0.01, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--learning-rate-actor', default=0.001, help="discrete actor  learning rate.", type=float)
    parser.add_argument('--learning-rate-actor-param', default=0.0001, help="parameter actor  learning rate.", type=float)
    parser.add_argument('--scale-actions', default=True, help="Scale actions.", type=bool)
    parser.add_argument('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
    parser.add_argument('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--split', default=False, help='Separate action-parameter inputs.', type=bool)
    parser.add_argument('--multipass', default=True,
                        help='Separate action-parameter inputs using multiple Q-network passes.',
                        type=bool)
    parser.add_argument('--indexed', default=False, help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False, help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False, help='Average weighted loss function.', type=bool)
    parser.add_argument('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--zero-index-gradients', default=False,
                        help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
                        type=bool)
    parser.add_argument('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
    parser.add_argument('--save-freq', default=2000, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save-dir', default="results/model/platform", help='Output directory.', type=str)
    parser.add_argument('--render-freq', default=1000, help='How often to render / save frames of an episode.',
                        type=int)
    parser.add_argument('--save-frames', default=False,
                        help="Save render frames from the environment. Incompatible with visualise.", type=bool)
    parser.add_argument('--visualise', default=False, help="Render game states. Incompatible with save-frames.",
                        type=bool)
    parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)
    args = parser.parse_args()
    for i in range(0, 5):
        args.seed = i
        run(args)
