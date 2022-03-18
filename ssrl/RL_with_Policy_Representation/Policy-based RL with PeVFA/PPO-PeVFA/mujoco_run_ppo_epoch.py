"""
Deterministic Direct Future Prediction
"""

import sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import time
import math
import datetime

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from networks.ppo import PPO
import scipy.signal


def evaluation(env, policy, ep_num, a_bound, max_ep_steps=1000):
    ep_reward_list = []

    for i in range(ep_num):
        s = env.reset()
        ep_reward = 0
        ep_step_count = 0
        t = 0
        t_inc = 1 / max_ep_steps

        for j in range(max_ep_steps):
            s = s.tolist()
            s.append(t)
            s = np.array(s)

            a = policy.choose_action(s)
            a = np.clip(a, -1, 1)
            scaled_a = np.multiply(a, a_bound)
            s_, r, done, info = env.step(scaled_a)

            t += t_inc
            ep_step_count += 1

            s = s_
            ep_reward += r

            if done:
                break

        ep_reward_list.append(ep_reward)
    return ep_reward_list

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def build_c_train_set(trajectories):
    observes = np.concatenate([t[0] for t in trajectories])
    disc_sum_rew = np.concatenate([t[3] for t in trajectories])
    # normalize advantages

    return observes, disc_sum_rew

def build_a_train_set(trajectories):
    observes = np.concatenate([t[0] for t in trajectories])
    actions = np.concatenate([t[1] for t in trajectories])
    advantages = np.concatenate([t[4] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages

def main(args):
    ###############################  training  ####################################
    seed = args.seed
    env_name = args.env
    train_interval = args.ti
    max_step = args.max_step
    lamb = args.lamb
    lr_a = args.lr_a

    k = args.k
    layer_num = args.layer_num
    is_save_data = args.is_save_data
    gpu_no = args.gpu_no

    gamma = 0.99

    #####################  hyper parameters  ####################

    MAX_TOTAL_STEPS = 1000 * max_step
    RANDOM_STEPS = 10000

    env = gym.make(env_name)
    env = env.unwrapped

    # FIXME 0520 - eval params
    env4eval = gym.make(env_name)
    env4eval = env4eval.unwrapped
    eval_step_point = 0
    num_eval_ep = 10
    eval_cnt = 0

    print('-- Env:', env_name)
    print('-- Seed:', seed)
    print('-- Configurations:', args)

    env.seed(seed)
    env4eval.seed(seed + 1)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no

    if gpu_no == '-1':
        sess = tf.Session()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


    # MAX_EP_STEPS = 1000
    # FIXME 0827
    MAX_EP_STEPS = 1000 if env_name != 'Reacher-v1' else 25
    t_inc = 1.0 / MAX_EP_STEPS

    s_dim = env.observation_space.shape[0] + 1
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    cur_policy = PPO(s_dim=s_dim, a_dim=a_dim, sess=sess,
                     gamma=gamma, k=k, batch_size=128,
                     lr_c=0.001, lr_a=lr_a,
                     c_epochs=args.c_epoch, a_epochs=args.a_epoch,
                     # FIXME 0827
                     # policy_logvar=0.0,
                     )
    
    a_loss_his, c_loss_his = [], []
    avg_a_loss_his, avg_c_loss_his = [], []
    # FIXME 1201
    return_his = []
    avg_return_his = []

    train_cnt = 0
    global_step_count = 0
    ep_num = 0

    # FIXME 1206 - new update interval
    update_ep_rcd = 0
    update_step_rcd = 0

    trajs = []

    while global_step_count < MAX_TOTAL_STEPS:
        s = env.reset()
        ep_reward = 0
        ep_s_traj, ep_a_traj, ep_r_traj = [], [], []

        ep_step_count = 0
        t = 0

        for j in range(MAX_EP_STEPS):
            s = s.tolist()
            s.append(t)
            s = np.array(s)

            a = cur_policy.choose_action(s)
            scaled_a = np.multiply(a, a_bound)
            s_, r, done, info = env.step(scaled_a)

            t += t_inc

            ep_s_traj.append(s)
            ep_a_traj.append(a)
            ep_r_traj.append(r)
            
            ep_step_count += 1
            global_step_count += 1

            s = s_
            ep_reward += r

            # FIXME 0520 - eval policy
            if global_step_count >= eval_step_point:
                eval_return_list = evaluation(env4eval, cur_policy, ep_num=num_eval_ep,
                                              a_bound=a_bound, max_ep_steps=MAX_EP_STEPS)
                cur_eval_return = sum(eval_return_list) / num_eval_ep
                avg_return_his.append(cur_eval_return)
                avg_c_loss = sum(c_loss_his[-5:]) / 5
                avg_a_loss = sum(a_loss_his[-5:]) / 5
                avg_c_loss_his.append(avg_c_loss)
                avg_a_loss_his.append(avg_a_loss)
                print('-----------------------------------------------------------')
                print('- Eval #', eval_cnt, 'total steps:', global_step_count, 'avg return:', cur_eval_return)
                eval_step_point += args.eval_interval
                eval_cnt += 1

            if done or global_step_count == MAX_TOTAL_STEPS:
                break

        # FIXME 1201
        return_his.append(ep_reward)

        traj = [np.array(ep_s_traj), np.array(ep_a_traj), np.array(ep_r_traj)]
        trajs.append(traj)

        # FIXME 1206 - new update inertval rule
        update_step_rcd += ep_step_count
        update_ep_rcd += 1
        # FIXME 0110
        # if update_ep_rcd == train_interval or update_step_rcd >= 2000:
        if update_ep_rcd == train_interval or update_step_rcd >= args.epoch_step:
            update_ep_rcd = 0
            update_step_rcd = 0
            for traj in trajs:
                t_s, t_a, t_r = traj
                t_dsc_r = discount(t_r, gamma)
                traj.append(t_dsc_r)

            observes, disc_sum_rew = build_c_train_set(trajs)
            c_loss = cur_policy.update_v(observes, disc_sum_rew)
            c_loss_his.append(c_loss)

            for traj in trajs:
                t_s, t_a, t_r, _ = traj
                t_v = cur_policy.predict_v(t_s)
                tds = t_r - t_v + np.append(t_v[1:] * gamma, 0)
                t_adv = discount(tds, gamma * lamb)

                traj.append(t_adv)

            observes, actions, advantages = build_a_train_set(trajs)
            a_loss = cur_policy.update_p(observes, actions, advantages)
            a_loss_his.append(a_loss)

            if train_cnt == 0 or (train_cnt + 1) % 5 == 0:
                avg_c_loss = sum(c_loss_his[-5:]) / 5
                avg_a_loss = sum(a_loss_his[-5:]) / 5
                # avg_c_loss_his.append(avg_c_loss)
                # avg_a_loss_his.append(avg_a_loss)

                avg_return = sum(return_his) / len(return_his)
                # avg_return_his.append(avg_return)
                return_his = []

                print('- Train cnt:', train_cnt + 1,
                      'Total steps:', global_step_count,
                      'avg_c_loss:', avg_c_loss,
                      'avg_a_loss:', avg_a_loss,
                      'avg_return:', avg_return,
                      )

            train_cnt += 1
            trajs = []

        ep_num += 1


    # FIXME 1027 - save loss trajs
    if is_save_data:
        print('=========================')
        print('- Saving data.')
        save_folder_path = './evaluation_data/ppo_' + str(lr_a) \
                           + '_c' + str(args.c_epoch) + 'a' + str(args.a_epoch) + 'es' + str(args.epoch_step) + '/'
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        file_index = save_folder_path + env_name + '_k' + str(k) + '_ln' + str(layer_num)
        if max_step not in [1000, 2000]:
            file_index += '_h' + str(max_step)
        file_index += '_s' + str(seed)
        # np.savez('./data/' + file_index,
        np.savez_compressed(file_index,
                            a_loss_trajs=avg_a_loss_his,
                            c_loss_trajs=avg_c_loss_his,
                            # prev_q_loss_trajs=avg_prev_c_loss_his,
                            # q_loss_after_trajs=avg_c_loss__his,
                            # prev_q_loss_after_trajs=avg_prev_c_loss__his,
                            avg_return=avg_return_his,
                            )
        print('- Data saved.')
        print('-------------------------')


def get_args_from_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help='random_seed')
    parser.add_argument("--env", type=str, default='HalfCheetah-v1', help='env')
    # parser.add_argument("--env", type=str, default='InvertedPendulum-v1', help='env')
    # parser.add_argument("--env", type=str, default='InvertedDoublePendulum-v1', help='env')
    # parser.add_argument("--env", type=str, default='Swimmer-v1', help='env')
    # parser.add_argument("--env", type=str, default='Reacher-v1', help='env')
    parser.add_argument("--max-step", type=int, default=2000, help='total_steps(k)')

    parser.add_argument("--lr-a", type=float, default=0.0001, help='learning_rate_of_actor')
    parser.add_argument("--ti", type=int, default=5, help='train_interval')
    parser.add_argument("--k", type=int, default=64, help='hidden_units_number')
    parser.add_argument("--layer-num", type=int, default=2, help='hidden_layer_num')
    parser.add_argument("--lamb", type=float, default=0.95, help='gae_lambda')
    parser.add_argument("--c-epoch", type=int, default=10, help='c_epoch')
    parser.add_argument("--a-epoch", type=int, default=10, help='a_epoch')
    parser.add_argument("--epoch-step", type=int, default=2000, help='a_epoch')

    parser.add_argument("--eval-interval", type=int, default=20000, help='number of steps per evaluation point')
    parser.add_argument("--is-save-data", type=bool, default=False, help='is_save_data')
    parser.add_argument("--is-plot-data", type=bool, default=False, help='is_plot_data')
    parser.add_argument("--gpu-no", type=str, default='-1', help='gpu_no')

    return parser.parse_args()

if __name__ == '__main__':

    t1 = time.time()
    arguments = get_args_from_parser()
    main(args=arguments)
    print('Running time: ', time.time() - t1)

