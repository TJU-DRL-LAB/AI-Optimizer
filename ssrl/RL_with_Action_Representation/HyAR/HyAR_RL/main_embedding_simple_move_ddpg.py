import numpy as np
import torch
import gym
import argparse
import os
from  HyAR_RL import utils
from agents import P_TD3_relable
from agents import P_DDPG_relable
import copy
from agents import OurDDPG
from agents import DDPG
# from sklearn.metrics import mean_squared_error
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from agents.pdqn_MPE_4_direction import PDQNAgent
from embedding import ActionRepresentation_vae
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def pad_action_(act, act_param):
    act_params = [0] * 4
    if act == 0:
        act_params[0] = act_param
        action = np.hstack(([5], act_params, [1], [0], [0], [0]))
    elif act == 1:
        act_params[1] = act_param
        action = np.hstack(([5], act_params, [0], [1], [0], [0]))
    elif act == 2:
        act_params[2] = act_param
        action = np.hstack(([5], act_params, [0], [0], [1], [0]))
    elif act == 3:
        act_params[3] = act_param
        action = np.hstack(([5], act_params, [0], [0], [0], [1]))

    return [action]


# A fixed seed is used for the eval environment
def evaluate(env, policy, action_rep, c_rate, max_steps, episodes=100):
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
            discrete_emb, parameter_emb = policy.select_action(state)
            true_parameter_emb = true_parameter_action(parameter_emb, c_rate)

            # select discrete action
            discrete_action_embedding = copy.deepcopy(discrete_emb)
            discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
            discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
            discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
            all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                      discrete_emb_1)
            parameter_action = all_parameter_action
            action = pad_action_(discrete_action, parameter_action)
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
    elif args.policy == "DDPG":
        kwargs["actor_lr"] = args.actor_lr
        kwargs["critic_lr"] = args.critic_lr
        kwargs["tau_actor"] = args.tau_actor
        kwargs["tau_critic"] = args.tau_critic
        policy = P_DDPG_relable.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embedding初始部分
    action_rep = ActionRepresentation_vae.Action_representation(state_dim=state_dim,
                                                                  action_dim=discrete_action_dim,
                                                                  parameter_action_dim=1,
                                                                  reduced_action_dim=discrete_emb_dim,
                                                                  reduce_parameter_action_dim=parameter_emb_dim
                                                                  )
    action_rep_target = copy.deepcopy(action_rep)
    replay_buffer = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                       parameter_action_dim=1,
                                       all_parameter_action_dim=parameter_action_dim,
                                       discrete_emb_dim=discrete_emb_dim,
                                       parameter_emb_dim=parameter_emb_dim,
                                       max_size=int(1e6))
    replay_buffer_embedding = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                                 parameter_action_dim=1,
                                                 all_parameter_action_dim=parameter_action_dim,
                                                 discrete_emb_dim=discrete_emb_dim,
                                                 parameter_emb_dim=parameter_emb_dim,
                                                 # max_size=int(1e7)
                                                 max_size=int(1e6)
                                                 )

    agent_pre = PDQNAgent(
        obs_shape_n, action_space=discrete_action_dim,
        parameter_action_dim=parameter_action_dim,
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
        actor_kwargs={'hidden_layers': [256, 256, 128, 64],
                      'action_input_layer': 0, },
        actor_param_kwargs={'hidden_layers': [256, 256, 128, 64],
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        zero_index_gradients=False,
        seed=args.seed)

    # ------Use random strategies to collect experience------
    max_steps = 30
    total_reward = 0.
    returns = []
    train_step = 0
    success = []
    for i in range(5000):
        state = obs_n
        state = np.array(state, dtype=np.float32, copy=False)[0]

        act, act_param, all_action_parameters = agent_pre.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent_pre.start_episode()
        flag = 0
        for j in range(max_steps):
            train_step += 1
            next_state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            if reward > 4:
                flag = 1
                done = True
            next_state = np.array(next_state, dtype=np.float32, copy=False)[0]
            next_act, next_act_param, next_all_action_parameters = agent_pre.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            state_next_state = next_state - state
            if act == 0:
                act_param_ = act_param[0]
            elif act == 1:
                act_param_ = act_param[1]
            elif act == 2:
                act_param_ = act_param[2]
            elif act == 3:
                act_param_ = act_param[3]


            replay_buffer_embedding.add(state, act, act_param_, all_action_parameters, discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=done)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward

            if done or j == max_steps - 1:
                obs_n = env.reset()
                break

        if flag == 1:
            success.append(1)
        else:
            success.append(0)

        returns.append(episode_reward)
        total_reward += episode_reward

        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f} success:{3:.4f}'.format(str(i), total_reward / (i + 1),
                                                                         np.array(returns[-100:]).mean(),
                                                                         np.array(success[-100:]).mean()))

    save_dir = "result/simple_move_model/kl_0.5/1.0/0527"
    save_dir = os.path.join(save_dir, "{}".format(str(48)))
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # ------VAE训练------
    initial_losses = []
    VAE_batch_size = 64
    vae_load_model = False
    vae_save_model = True
    # vae_load_model = True
    # vae_save_model = False
    if vae_load_model:
        print("load model")
        title = "vae" + "{}".format(str(4000))
        action_rep.load(title, save_dir)
        print("load discrete embedding", action_rep.discrete_embedding())
    print("pre VAE training phase started...")
    recon_s_loss = []
    c_rate, recon_s = vae_train(action_rep=action_rep, train_step=5000, replay_buffer=replay_buffer_embedding,
                                batch_size=VAE_batch_size,
                                save_dir=save_dir, vae_save_model=vae_save_model, embed_lr=1e-4)
    recon_s_loss.append(recon_s)
    print("discrete embedding", action_rep.discrete_embedding())
    print("c_rate", c_rate)
    print("recon_s", recon_s)
    # -------TD3训练------
    print("TD3 train")
    total_reward = 0.
    returns = []
    Reward = []
    Reward_100 = []
    Test_Reward = []
    max_steps = 30
    cur_step = 0
    flag = 0
    Test_success = []
    mse_state = []
    Test_epioside_step = []
    internal = 100
    total_timesteps = 0
    t = 0
    discrete_relable_rate, parameter_relable_rate = 0, 0
    # for t in range(int(args.max_episodes)):
    while total_timesteps < args.max_timesteps:

        state = obs_n
        state = np.array(state, dtype=np.float32, copy=False)[0]

        discrete_emb, parameter_emb = policy.select_action(state)
        # 探索
        if t < args.epsilon_steps:
            epsilon = args.expl_noise_initial - (args.expl_noise_initial - args.expl_noise) * (
                    t / args.epsilon_steps)
        else:
            epsilon = args.expl_noise


        discrete_emb = (
                discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
        ).clip(-max_action, max_action)
        parameter_emb = (
                parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
        ).clip(-max_action, max_action)

        # parameter_emb = parameter_emb * c_rate
        true_parameter_emb = true_parameter_action(parameter_emb, c_rate)

        # select discrete action
        discrete_action_embedding = copy.deepcopy(discrete_emb)
        discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
        discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
        discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
        all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                  discrete_emb_1)
        parameter_action = all_parameter_action
        action = pad_action_(discrete_action, parameter_action)

        episode_reward = 0.
        flag = 0
        if cur_step >= args.start_timesteps:
            discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate, recon_s,
                                                                         args.batch_size)
        for i in range(max_steps):
            total_timesteps += 1
            next_state, reward, done_n, _ = env.step(action)
            done = all(done_n)
            reward = reward[0]
            if reward > 4:
                flag = 1
                done = True

            next_state = np.array(next_state, dtype=np.float32, copy=False)[0]
            cur_step = cur_step + 1

            state_next_state = next_state - state
            replay_buffer.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                              all_parameter_action=None,
                              discrete_emb=discrete_emb,
                              parameter_emb=parameter_emb,
                              next_state=next_state,
                              state_next_state=state_next_state,
                              reward=reward, done=done)
            replay_buffer_embedding.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                                        all_parameter_action=None,
                                        discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=done)
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
            # if t % 100 == 0:
            #     print("真实动作", next_discrete_action, next_all_parameter_action)
            # env.render()
            next_parameter_action = next_all_parameter_action
            next_action = pad_action_(next_discrete_action, next_parameter_action)

            discrete_emb, parameter_emb, action, discrete_action, parameter_action = next_discrete_emb, next_parameter_emb, next_action, next_discrete_action, next_parameter_action

            state = next_state
            if cur_step >= args.start_timesteps:
                discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate, recon_s,
                                                                             args.batch_size)
            # if t % 100 == 0:
            #     print("discrete_relable_rate,parameter_relable_rate", discrete_relable_rate, parameter_relable_rate)
            if total_timesteps % args.eval_freq == 0:
                print('{0:5s} R:{1:.4f} r100:{2:.4f} success:{3:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                                             np.array(returns[-100:]).mean(),
                                                                             np.array(success[-100:]).mean()))
                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())
                Test_Reward_50, Test_success_rate, Test_epioside_step_50 = evaluate(env, policy, action_rep, c_rate,
                                                                                    max_steps=30, episodes=50)
                Test_Reward.append(Test_Reward_50)
                Test_success.append(Test_success_rate)
                Test_epioside_step.append(Test_epioside_step_50)

            episode_reward += reward

            if done or i == max_steps - 1:
                obs_n = env.reset()
                break
        t += 1
        returns.append(episode_reward)
        total_reward += episode_reward
        if flag == 1:
            success.append(1)
        else:
            success.append(0)

        # if t % 100 == 0:
        #     print('{0:5s} R:{1:.4f} r100:{2:.4f} success:{3:.4f}'.format(str(t), total_reward / (t + 1),
        #                                                                  np.array(returns[-100:]).mean(),
        #                                                                  np.array(success[-100:]).mean()))
        #     Reward.append(total_reward / (t + 1))
        #     Reward_100.append(np.array(returns[-100:]).mean())
        #     Test_Reward_50, Test_success_rate, Test_epioside_step_50= evaluate(env, policy, action_rep, c_rate, max_steps=30, episodes=50)
        #     Test_Reward.append(Test_Reward_50)
        #     Test_success.append(Test_success_rate)
        #     Test_epioside_step.append(Test_epioside_step_50)

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
    dir = "result/ddpg/simple_move"
    data = "0820"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)

    # title1 = "Reward_td3_simple_move_4_direction_embedding_nopre_relable_"
    title2 = "Reward_100_ddpg_simple_move_4_direction_embedding_nopre_relable_"
    title3 = "Test_Reward_ddpg_simple_move_4_direction_embedding_nopre_relable_"
    title4 = "Test_success_ddpg_simple_move_4_direction_embedding_nopre_relable_"
    title5 = "Test_epioside_step_ddpg_simple_move_4_direction_embedding_nopre_relable_"
    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_success, delimiter=',')
    np.savetxt(os.path.join(redir, title5 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step, delimiter=',')


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
        if vae_save_model:
            if counter % 1000 == 0 and counter >= 1000:
                title = "vae" + "{}".format(str(counter))
                action_rep.save(title, save_dir)
                print("vae save model")

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
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
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
    parser.add_argument("--actor_lr", default=1e-4, type=int)
    parser.add_argument("--critic_lr", default=1e-3, type=int)
    parser.add_argument("--tau_actor", default=0.005, type=int)
    parser.add_argument("--tau_critic", default=0.005, type=int)
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)
