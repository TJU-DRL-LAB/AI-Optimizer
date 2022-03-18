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

from networks.ppo_pevf_e2e_opr1 import PPO_PEVF
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
    lr_pc = args.lr_pc
    lr_a = args.lr_a

    k = args.k
    prev_type = args.prev_type
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


    MAX_EP_STEPS = 1000
    t_inc = 1.0 / MAX_EP_STEPS

    s_dim = env.observation_space.shape[0] + 1
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    # pair_len = args.pair_len
    # train_num = args.train_num
    cur_policy = PPO_PEVF(s_dim=s_dim, a_dim=a_dim, pr_dim=args.pr_dim, sess=sess,
                          gamma=gamma, k=k, prev_type=prev_type, batch_size=128,
                          lr_c=0.001, lr_a=lr_a, lr_pc=lr_pc,
                          c_epochs=args.c_epoch, a_epochs=args.a_epoch,
                          # FIXME 0522
                          memory_size=200000, policy_size=10000,
                          # memory_size=200000, policy_size=100000,
                          # memory_size=1000000, policy_size=100000,
                          # memory_size=500000, policy_size=100000,
                          # memory_size=100000, policy_size=100000,
                          pr_model=None)
    
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
                                              a_bound=a_bound, max_ep_steps=1000)
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

        if global_step_count > 5000:
            update_num = global_step_count // args.update_freq - (global_step_count - ep_step_count) // args.update_freq
            for _ in range(update_num):
                cur_policy.update_pevf(batch_size=args.pevf_batch_size)

        # FIXME 1201
        return_his.append(ep_reward)

        traj = [np.array(ep_s_traj), np.array(ep_a_traj), np.array(ep_r_traj)]
        trajs.append(traj)

        # FIXME 1206 - new update inertval rule
        update_step_rcd += ep_step_count
        update_ep_rcd += 1
        # FIXME 0110
        # if update_ep_rcd == train_interval or update_step_rcd >= 1000:
        if update_ep_rcd == train_interval or update_step_rcd >= 2000:
            update_ep_rcd = 0
            update_step_rcd = 0

            # FIXME 1029 - store transitions
            # FIXME 0508 - calculate policy params
            cur_params = cur_policy.get_params()
            policy_idx = cur_policy.store_policy_data(cur_params)
            for traj in trajs:
                s_traj = traj[0]
                r_traj = traj[2]
                dsr_re_traj = discount(r_traj, gamma=gamma)
                for iot in range(len(s_traj)):
                    cur_s = s_traj[iot]
                    cur_dsc_re = dsr_re_traj[iot]
                    cur_policy.store_transition(cur_s.tolist(), cur_dsc_re, policy_idx)

            for traj in trajs:
                t_s, t_a, t_r = traj
                t_dsc_r = discount(t_r, gamma)
                traj.append(t_dsc_r)

            observes, disc_sum_rew = build_c_train_set(trajs)
            pevf_c_loss = cur_policy.update_v(observes, disc_sum_rew, cur_params)
            c_loss_his.append(pevf_c_loss)

            for traj in trajs:
                t_s, t_a, t_r, _ = traj
                t_params = np.tile(cur_params, [t_s.shape[0], 1])
                t_v = cur_policy.predict_v_prev(t_s, t_params)
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
        save_folder_path = './evaluation_data/ppo_pevf_e2e_opr_Sep'
        save_folder_path += '_pd' + str(args.pr_dim)
        save_folder_path += '_freq' + str(args.update_freq)
        save_folder_path += '_c' + str(args.c_epoch) + 'a' + str(args.a_epoch)
        save_folder_path += '_' + str(lr_a) + '/'
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        file_index = save_folder_path + env_name \
                     + '_M200kP10k' \
                     + '_pbs' + str(args.pevf_batch_size) \
                     + '_pt' + str(prev_type) + '_pclr' + str(lr_pc)
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

    parser.add_argument("--env", type=str, default='HalfCheetah-v1', help='env')
    # parser.add_argument("--env", type=str, default='InvertedPendulum-v1', help='env')
    # parser.add_argument("--env", type=str, default='InvertedDoublePendulum-v1', help='env')
    # parser.add_argument("--env", type=str, default='Swimmer-v1', help='env')
    parser.add_argument("--seed", type=int, default=1, help='random_seed')
    parser.add_argument("--max-step", type=int, default=2000, help='total_steps(k)')

    parser.add_argument("--lr-pc", type=float, default=0.001, help='learning_rate_of_PEVF')
    parser.add_argument("--prev-type", type=int, default=0, help='structure_of_PREV')
    parser.add_argument("--update-freq", type=int, default=10, help='update_frequency_of_PEVF')
    parser.add_argument("--pevf-batch-size", type=int, default=64, help='batch_size_of_PEVF')
    parser.add_argument("--pr-dim", type=int, default=64, help='pr_dim_of_PEVF')

    parser.add_argument("--lr-a", type=float, default=0.0001, help='learning_rate_of_actor')
    parser.add_argument("--ti", type=int, default=5, help='train_interval')
    parser.add_argument("--k", type=int, default=64, help='hidden_units_number')
    parser.add_argument("--layer-num", type=int, default=2, help='hidden_layer_num')
    parser.add_argument("--lamb", type=float, default=0.95, help='gae_lambda')
    parser.add_argument("--c-epoch", type=int, default=10, help='c_epoch')
    parser.add_argument("--a-epoch", type=int, default=10, help='a_epoch')

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

