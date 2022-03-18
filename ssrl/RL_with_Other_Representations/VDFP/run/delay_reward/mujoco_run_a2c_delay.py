"""
    run A2C in MuJoCo tasks under delay reward settings

"""

import sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import time

sys.path.append('./')
sys.path.append('../')

from networks.a2c import A2C
import scipy.signal

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# print(os.environ.get('MUJOCO_PY_MJKEY_PATH'))
# print(os.environ.get('MUJOCO_PY_MJPRO_PATH'))

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t[0] for t in trajectories])
    actions = np.concatenate([t[1] for t in trajectories])
    disc_sum_rew = np.concatenate([t[4] for t in trajectories])
    advantages = np.concatenate([t[5] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew

#####################  hyper parameters  ####################

MAX_TOTAL_STEPS = 2000000
RANDOM_STEPS = 10000

RENDER = False
# RENDER = True

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default='HalfCheetah-v1', help='env')
parser.add_argument("--delay_type", type=int, default=1, help='delay_reward_type')

parser.add_argument("--seed", type=int, default=111, help='random_seed')
parser.add_argument("--ti", type=int, default=2, help='train_interval')
# when lamb = 0.0, it equals the vanilla A2C w/o GAE
parser.add_argument("--lamb", type=float, default=0, help='gae_lambda')
parser.add_argument("--delay_step", type=int, default=16, help='delay_step')

args = parser.parse_args()
seed = args.seed

env_name = args.env
delay_type = args.delay_type

train_interval = args.ti
lamb = args.lamb
delay_step = args.delay_step

sess = tf.Session()
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# sess = tf.Session(config=config)

env = gym.make(env_name)
env = env.unwrapped

print('-- Env:', env_name)
print('-- Seed:', seed)

env.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

MAX_EP_STEPS = 1000
t_inc = 1.0 / MAX_EP_STEPS

s_dim = env.observation_space.shape[0] + 1
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

a2c = A2C(s_dim, a_dim, sess, gamma=0.99, lr_c=0.001, lr_a=0.0001)

# --------------------------------- summary operations ------------------------------
# IS_SAVE_LOG = True
IS_SAVE_LOG = False
if IS_SAVE_LOG:
    var3 = tf.placeholder(tf.float32, None, name='episode_reward')
    tf.summary.scalar('episode_reward', var3)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tf_logs/' + env_name + 'delayed' + str(delay_type) + '_' + str(delay_step)
                                   + '/a2c/gamma99_lamd' + str(lamb * 100)
                                   + 'lr0001x10_ti' + str(train_interval) + '/seed' + str(seed), sess.graph)

t1 = time.time()

train_count = 0
global_step_count = 0

c_loss_his, a_loss_his = [], []


ep_num = 0

trajs = []

while global_step_count < MAX_TOTAL_STEPS:
    s = env.reset()
    ep_reward = 0

    obs, acts, rews = [], [], []

    ep_step_count = 0
    t = 0

    rewards = []

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
            time.sleep(0.03)

        s = s.tolist()
        s.append(t)
        s = np.array(s)

        a = a2c.choose_action(s)
        scaled_a = np.multiply(a, a_bound)
        s_, r, done, info = env.step(scaled_a)

        rewards.append(r)

        if delay_type == 1:
            if (ep_step_count + 1) % delay_step == 0 or ep_step_count == MAX_EP_STEPS or done:
                r_exp = sum(rewards)
                rewards = []
            else:
                r_exp = 0.0
        else:
            if ep_step_count == MAX_EP_STEPS or done:
                if delay_step + 1 > len(rewards):
                    r_exp = sum(rewards)
                else:
                    r_exp = sum(rewards[-delay_step - 1:])
            else:
                if delay_step + 1 > len(rewards):
                    r_exp = 0.0
                else:
                    r_exp = rewards[-delay_step - 1]

        t += t_inc

        obs.append(s)
        acts.append(a)
        rews.append(r_exp)
        # rews.append(r)

        ep_step_count += 1
        global_step_count += 1

        s = s_
        ep_reward += r

        if done or global_step_count == MAX_TOTAL_STEPS:
            break

    traj = [np.array(obs), np.array(acts), np.array(rews)]
    trajs.append(traj)

    # TODO
    if (ep_num + 1) % train_interval == 0:
        buffer = []
        for traj in trajs:
            t_s, t_a, t_r = traj
            t_v = a2c.predict_v(t_s)
            t_dsc_r = discount(t_r, 0.99)
            tds = t_r - t_v + np.append(t_v[1:] * 0.99, 0)
            t_adv = discount(tds, 0.99 * lamb)
            # t_adv = discount(tds, 0.99 * 0.0)

            traj.append(t_v)
            traj.append(t_dsc_r)
            traj.append(t_adv)

        observes, actions, advantages, disc_sum_rew = build_train_set(trajs)
        a_loss = a2c.update_p(observes, actions, advantages)
        c_loss = a2c.update_v(observes, disc_sum_rew)

        trajs = []
        train_count += 1


    print('- Episode:', ep_num, ' Reward: %i' % int(ep_reward),
          'Total steps:', global_step_count,
          'Train count:', train_count,
          # 'Loss:', sum(ae_loss_his[-100:]) / 100.0, sum(p_loss_his[-100:]) / 100.0, sum(a_loss_his[-100:]) / 100.0
          )

    ep_num += 1

    if IS_SAVE_LOG:
        summary = sess.run(merged, feed_dict={var3: ep_reward})
        writer.add_summary(summary, global_step_count)


print('Running time: ', time.time() - t1)

