import numpy as np
import torch
import gym
import argparse
import os
from  HyAR_RL import utils
from agents import P_TD3_relable
from agents import P_DDPG_relable
import copy
# import OurDDPG
# import DDPG
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
import matplotlib.pyplot as plt
from agents.pdqn_hard_goal import PDQNAgent
from agents.utils import soft_update_target_network, hard_update_target_network
from embedding import ActionRepresentation_vae
import torch
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def pad_action_(act, act_param):
    c_rate = [[-1.0, -0.6], [-0.6, -0.2], [0.2, 0.2], [0.2, 0.6], [0.6, 1.0]]
    # print("c_rate",c_rate[0])
    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    if act == 0:
        params[0][0] = act_param[0]
        params[0][1] = act_param[1]
    elif act == 1:
        act_param=true_action(0, act_param, c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 2:
        act_param=true_action(1, act_param, c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 3:
        act_param=true_action(2, act_param, c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 4:
        act_param=true_action(3, act_param, c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1
    elif act == 5:
        act_param=true_action(4, act_param, c_rate)
        act_param=np.array([act_param])
        params[1] = act_param
        act = 1

    elif act == 6:
        act_param=true_action(0, act_param, c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 7:
        act_param=true_action(1, act_param, c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 8:
        act_param=true_action(2, act_param, c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 9:
        act_param=true_action(3, act_param, c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    elif act == 10:
        act_param=true_action(4, act_param, c_rate)
        act_param=np.array([act_param])
        params[2] = act_param
        act = 2
    return (act, params)

# A fixed seed is used for the eval environment
def evaluate(env, policy, action_rep, c_rate, episodes=100):
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
            discrete_emb, parameter_emb = policy.select_action(state)
            true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
            # parameter_emb = parameter_emb * c_rate
            # select discrete action
            discrete_action_embedding = copy.deepcopy(discrete_emb)
            discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
            discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
            discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
            all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                      discrete_emb_1)
            if discrete_action == 0 :
                all_parameter_action = all_parameter_action
            else:
                all_parameter_action = all_parameter_action[0]
            parameter_action = all_parameter_action
            action = pad_action_(discrete_action, parameter_action)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward

        epioside_steps.append(t)
        returns.append(total_reward)
    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} success: {(np.array(returns) == 50.).sum() / len(returns):.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean(), (
            np.array(returns) == 50.).sum() / len(returns)


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
        env = gym.make(args.env)
        env = ScaledStateWrapper(env)
        initial_params_ = [3., 10., 400.]
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                    env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        env = PlatformFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
    elif args.env == "Goal-v0":
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
    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.spaces[0].shape[0]

    discrete_action_dim = 11
    parameter_action_dim = 12

    discrete_emb_dim = 6
    parameter_emb_dim = 6
    max_action = 1.0
    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)

    kwargs = {
        "state_dim": state_dim,
        "discrete_action_dim": discrete_emb_dim,
        "parameter_action_dim": parameter_emb_dim,
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
        policy = P_TD3_relable.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embedding初始部分
    action_rep = ActionRepresentation_vae.Action_representation(state_dim=state_dim,
                                                                  action_dim=discrete_action_dim,
                                                                  parameter_action_dim=2,
                                                                  reduced_action_dim=discrete_emb_dim,
                                                                  reduce_parameter_action_dim=parameter_emb_dim
                                                                  )

    replay_buffer = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                       parameter_action_dim=2,
                                       all_parameter_action_dim=parameter_action_dim,
                                       discrete_emb_dim=discrete_emb_dim,
                                       parameter_emb_dim=parameter_emb_dim,
                                       max_size=int(1e5))

    replay_buffer_embedding = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                                 parameter_action_dim=2,
                                                 all_parameter_action_dim=parameter_action_dim,
                                                 discrete_emb_dim=discrete_emb_dim,
                                                 parameter_emb_dim=parameter_emb_dim,
                                                 # max_size=int(2e7)
                                                 max_size=int(2e6)
                                                 )

    agent_pre = PDQNAgent(
        env.observation_space.spaces[0], action_space = discrete_action_dim,
        parameter_action_dim = parameter_action_dim,
        batch_size=128,
        learning_rate_actor=0.001,
        learning_rate_actor_param=0.0001,
        epsilon_steps=1000,
        gamma=0.9,
        tau_actor=0.1,
        tau_actor_param=0.01,
        clip_grad=10.,
        indexed=False,
        weighted=False,
        average=False,
        random_weighted=False,
        initial_memory_threshold=500,
        use_ornstein_noise=False,
        replay_memory_size=10000,
        epsilon_final=0.01,
        inverting_gradients=True,
        zero_index_gradients=False,
        seed=args.seed)

    # ------Use random strategies to collect experience------

    max_steps = 150
    total_reward = 0.
    returns = []
    for i in range(20000):

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent_pre.act(state)
        action = pad_action(act, act_param)
        episode_reward = 0.
        agent_pre.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent_pre.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            # print("next_action",next_act,next_action)
            state_next_state = next_state - state
            # print(act_param)
            if act == 0 :
                act_param=act_param[0:2]
            else:
                act_param=act_param[act+1:act+2]
                act_param = np.append(act_param, 0.)
            # print("act_param",act, act_param)
            replay_buffer_embedding.add(state, act, act_param, all_action_parameters, discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=terminal)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            if terminal:
                break
        # agent_pre.end_episode()
        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('per-train-{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1),
                                                                   np.array(returns[-100:]).mean()))
    s_dir = "result/goal_model"
    save_dir = os.path.join(s_dir, "{}".format(str("embedding")))
    save_dir_rl = os.path.join(s_dir, "{}".format(str("policy")))
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_rl, exist_ok=True)
    rl_save_model=True
    rl_load_model=False

    # rl_save_model=False
    # rl_load_model=True
    # ------VAE训练------
    VAE_batch_size = 64
    vae_save_model = True
    vae_load_model = False

    # vae_save_model = False
    # vae_load_model = True
    if vae_load_model:
        print("embedding load model")
        title = "vae" + "{}".format(str(4000))
        action_rep.load(title, save_dir)
        # print("load discrete embedding", action_rep.discrete_embedding())
    # print("pre VAE training phase started...")
    recon_s_loss = []
    c_rate, recon_s = vae_train(action_rep=action_rep, train_step=5000, replay_buffer=replay_buffer_embedding,
                                batch_size=VAE_batch_size,
                                save_dir=save_dir, vae_save_model=vae_save_model, embed_lr=1e-4)

    print("c_rate", c_rate)
    print("discrete embedding", action_rep.discrete_embedding())

    # -------TD3训练------
    print("TD3 train")
    total_reward = 0.
    returns = []
    Reward = []
    Reward_100 = []
    Test_Reward_100 = []
    Test_epioside_step_100 = []
    Test_success_rate_100 = []
    Crate_all=[]
    max_steps = 150
    cur_step = 0
    internal = 10
    total_timesteps = 0
    t = 0
    discrete_relable_rate, parameter_relable_rate = 0, 0
    # for t in range(int(args.max_episodes)):
    # if rl_load_model:
    #     title = "td3" + "{}".format(str(100000))
    #     policy.load(title, save_dir_rl)
    #     print("rl load model")

    while total_timesteps < args.max_timesteps:

        # if rl_save_model:
        #     # if i % 1000 == 0 and i >= 1000:
        #     if total_timesteps % 10000 == 0 :
        #         title = "td3" + "{}".format(str(total_timesteps))
        #         policy.save(title, save_dir_rl)
        #         print("rl save model")
        #         title = "vae" + "{}".format(str(total_timesteps))
        #         action_rep.save(title, save_dir)
        #         print("embedding save model")
        #         c_rate=np.array(c_rate)
        #         print(c_rate)
        #         np.save(os.path.join(save_dir, "crate" + "{}".format(str(total_timesteps) + ".npy")), c_rate)
        #
        #         # ar_load = np.load(os.path.join(save_dir, "crate" + "{}".format(str(total_timesteps) + ".npy")))

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        discrete_emb, parameter_emb = policy.select_action(state)
        # 探索
        if t < args.epsilon_steps:
            epsilon = args.expl_noise_initial - (args.expl_noise_initial - args.expl_noise) * (
                    t / args.epsilon_steps)
        else:
            epsilon = args.expl_noise

        if rl_load_model:
            epsilon=0.0

        discrete_emb = (
                discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
        ).clip(-max_action, max_action)
        parameter_emb = (
                parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
        ).clip(-max_action, max_action)
        true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
        # parameter_emb = parameter_emb * c_rate


        # select discrete action
        discrete_action_embedding = copy.deepcopy(discrete_emb)
        discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
        discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
        discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()

        all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                  discrete_emb_1)

        if discrete_action ==0 :
            all_parameter_action = all_parameter_action
        else:
            all_parameter_action = all_parameter_action[0]
        parameter_action = all_parameter_action
        # print("discrete_action, parameter_action",discrete_action, parameter_action)
        action = pad_action_(discrete_action, parameter_action)
        episode_reward = 0.

        if discrete_action == 0 :
            parameter_action = parameter_action
        else:
            parameter_action = np.append(parameter_action, 0.)

        if cur_step >= args.start_timesteps:
            discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate,
                                                                         recon_s,
                                                                         args.batch_size)
        # if t % 100 == 0:
        #     print("discrete_relable_rate, parameter_relable_rate", discrete_relable_rate, parameter_relable_rate)
        for i in range(max_steps):

            cur_step = cur_step + 1
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            # reward_scale
            # r = reward * reward_scale
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            state_next_state = next_state - state
            replay_buffer.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                              all_parameter_action=None,
                              discrete_emb=discrete_emb,
                              parameter_emb=parameter_emb,
                              next_state=next_state,
                              state_next_state=state_next_state,
                              reward=reward, done=terminal)
            replay_buffer_embedding.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                                        all_parameter_action=None,
                                        discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=terminal)

            next_discrete_emb, next_parameter_emb = policy.select_action(next_state)

            # if t % 100 == 0:
            #     print("策略输出", next_discrete_emb, next_parameter_emb)
            next_discrete_emb = (
                    next_discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
            ).clip(-max_action, max_action)
            next_parameter_emb = (
                    next_parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
            ).clip(-max_action, max_action)
            # next_parameter_emb = next_parameter_emb * c_rate
            true_next_parameter_emb = true_parameter_action(next_parameter_emb, c_rate)
            # select discrete action
            next_discrete_action_embedding = copy.deepcopy(next_discrete_emb)
            next_discrete_action_embedding = torch.from_numpy(next_discrete_action_embedding).float().reshape(1, -1)
            next_discrete_action = action_rep.select_discrete_action(next_discrete_action_embedding)
            next_discrete_emb_1 = action_rep.get_embedding(next_discrete_action).cpu().view(-1).data.numpy()
            # select parameter action
            next_all_parameter_action = action_rep.select_parameter_action(next_state, true_next_parameter_emb,
                                                                           next_discrete_emb_1)
            if t % 100 == 0:
                print("真实动作", next_discrete_action, next_all_parameter_action)
            # env.render()

            if next_discrete_action == 0 :
                next_all_parameter_action = next_all_parameter_action
            else:
                next_all_parameter_action = next_all_parameter_action[0]
            next_parameter_action = next_all_parameter_action
            next_action = pad_action_(next_discrete_action, next_parameter_action)
            if t % 100 == 0:
                print("env动作", next_action)
            if next_discrete_action == 0 :
                next_parameter_action = next_parameter_action
            else:
                next_parameter_action = np.append(next_parameter_action, 0.)
            discrete_emb, parameter_emb, action, discrete_action, parameter_action = next_discrete_emb, next_parameter_emb, next_action, next_discrete_action, next_parameter_action
            state = next_state
            if cur_step >= args.start_timesteps:
                discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate,
                                                                             recon_s,
                                                                             args.batch_size)
            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                           np.array(returns[-100:]).mean()))

                while not terminal:
                    state = np.array(state, dtype=np.float32, copy=False)
                    discrete_emb, parameter_emb = policy.select_action(state)
                    true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
                    discrete_action_embedding = copy.deepcopy(discrete_emb)
                    discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
                    discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
                    discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
                    all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                              discrete_emb_1)
                    if discrete_action == 0 :
                        all_parameter_action = all_parameter_action
                    else:
                        all_parameter_action = all_parameter_action[0]
                    parameter_action = all_parameter_action
                    action = pad_action_(discrete_action, parameter_action)
                    (state, _), reward, terminal, _ = env.step(action)

                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())

                Test_Reward, Test_epioside_step, Test_success_rate = evaluate(env, policy, action_rep, c_rate,
                                                                              episodes=100)
                Test_Reward_100.append(Test_Reward)
                Test_epioside_step_100.append(Test_epioside_step)
                Test_success_rate_100.append(Test_success_rate)

            if terminal:
                break

        t = t + 1
        returns.append(episode_reward)
        total_reward += episode_reward


        # vae 训练
        # if t % 1000 == 0 and t >= 1000:
        if t % internal == 0 and t >= 1000:
            # print("表征调整")
            # print("vae train")
            c_rate, recon_s = vae_train(action_rep=action_rep, train_step=1, replay_buffer=replay_buffer_embedding,
                                        batch_size=VAE_batch_size, save_dir=save_dir, vae_save_model=vae_save_model,
                                        embed_lr=1e-4)
            recon_s_loss.append(recon_s)
            # print("discrete embedding", action_rep.discrete_embedding())
            # print("c_rate", c_rate)
            # print("recon_s", recon_s)
    print("save txt")
    dir = "result/TD3/goal"
    data = "0829"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    # title1 = "Reward_td3_goal_embedding_nopre_"
    title2 = "Reward_100_td3_hard_goal_embedding_nopre_"
    title3 = "Test_Reward_100_td3_hard_goal_embedding_nopre_"
    title4 = "Test_epioside_step_100_td3_hard_goal_embedding_nopre_"
    title5 = "Test_success_rate_100_td3_hard_goal_embedding_nopre_"

    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step_100,
               delimiter=',')
    np.savetxt(os.path.join(redir, title5 + "{}".format(str(args.seed) + ".csv")), Test_success_rate_100,
               delimiter=',')


def vae_train(action_rep, train_step, replay_buffer, batch_size, save_dir, vae_save_model, embed_lr):
    initial_losses = []
    for counter in range(int(train_step) + 10):
        losses = []
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = action_rep.unsupervised_loss(state,
                                                                                     discrete_action.reshape(1,
                                                                                                             -1).squeeze().long(),
                                                                                     parameter_action,
                                                                                     state_next_state,
                                                                                     batch_size, embed_lr)
        losses.append(vae_loss)
        initial_losses.append(np.mean(losses))

        if counter % 100 == 0 and counter >= 100:
            # print("load discrete embedding", action_rep.discrete_embedding())
            print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))

        # Terminate initial phase once action representations have converged.
        if len(initial_losses) >= train_step and np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:]):
            # print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            # print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            # print("Converged...", len(initial_losses))
            break
        # if vae_save_model:
        #     if counter % 1000 == 0 and counter >= 1000:
        #         title = "vae" + "{}".format(str(counter))
        #         action_rep.save(title, save_dir)
        #         print("embedding save model")

    state_, discrete_action_, parameter_action_, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state_, reward, not_done = replay_buffer.sample(
        batch_size=5000)
    c_rate, recon_s = action_rep.get_c_rate(state_, discrete_action_.reshape(1, -1).squeeze().long(), parameter_action_,
                                            state_next_state_, batch_size=5000, range_rate=2)
    return c_rate, recon_s


def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2
    offset = c_rate[0] - 1 * median
    return median, offset


def true_parameter_action(parameter_action, c_rate):
    parameter_action_ = copy.deepcopy(parameter_action)
    for i in range(len(parameter_action)):
        median, offset = count_boundary(c_rate[i])
        parameter_action_[i] = parameter_action_[i] * median + offset
    return parameter_action_


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="P-TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='Goal-v0')  # platform goal HFO
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=128, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_episodes", default=50000, type=int)  # Max time steps to run environment
    parser.add_argument("--max_embedding_episodes", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=300000, type=float)  # Max time steps to run environment for

    parser.add_argument("--epsilon_steps", default=1000, type=int)  # Max time steps to epsilon environment
    parser.add_argument("--expl_noise_initial", default=1.0)  # Std of Gaussian exploration noise 1.0
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise 0.1

    parser.add_argument("--relable_steps", default=1000, type=int)  # Max time steps relable
    parser.add_argument("--relable_initial", default=1.0)  #
    parser.add_argument("--relable_final", default=0.0)  #

    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.1)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
