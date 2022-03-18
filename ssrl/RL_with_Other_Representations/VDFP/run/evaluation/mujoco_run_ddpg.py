"""
    run DDPG in MuJoCo tasks

"""

import sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import time

sys.path.append('./')
sys.path.append('../')

from networks.ddpg import DDPG

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# print(os.environ.get('MUJOCO_PY_MJKEY_PATH'))
# print(os.environ.get('MUJOCO_PY_MJPRO_PATH'))


#####################  hyper parameters  ####################

MAX_TOTAL_STEPS = 2000000
RANDOM_STEPS = 10000

RENDER = False
# RENDER = True

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default='HalfCheetah-v1', help='env')
parser.add_argument("--seed", type=int, default=111, help='random_seed')
parser.add_argument("--ti", type=int, default=1, help='train_interval')

args = parser.parse_args()
seed = args.seed

env_name = args.env
train_interval = args.ti

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

ddpg = DDPG(s_dim, a_dim, sess, gamma=0.99, lr_c=0.001, lr_a=0.0001, tau=0.001, memory_size=100000)

# --------------------------------- summary operations ------------------------------
# IS_SAVE_LOG = True
IS_SAVE_LOG = False
if IS_SAVE_LOG:
    var3 = tf.placeholder(tf.float32, None, name='episode_reward')
    tf.summary.scalar('episode_reward', var3)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tf_logs/' + env_name
                                   + '/ddpg_fixed_noise/var01_gamma99_lr0001x10_tau1e3_ti'
                                   + str(train_interval)
                                   +'/seed' + str(seed), sess.graph)

t1 = time.time()

train_count = 0
global_step_count = 0

ae_loss_his, p_loss_his, a_loss_his = [], [], []

ep_num = 0

while global_step_count < MAX_TOTAL_STEPS:
    s = env.reset()
    ep_reward = 0

    ep_step_count = 0
    t = 0

    for j in range(MAX_EP_STEPS):
        # FIXME 0114 add time-varying feature
        s = s.tolist()
        s.append(t)
        s = np.array(s)

        a = ddpg.choose_action(s)
        noise = np.random.normal(0, 0.1, size=a_dim)
        a = np.clip(a + noise, -1, 1)

        scaled_a = np.multiply(a, a_bound)   # add randomness to action selection for exploration
        s_, r, done, info = env.step(scaled_a)

        t += t_inc

        l_s_ = s_.tolist()
        l_s_.append(t)

        ddpg.store_transition(s.tolist(), a.tolist(), r, l_s_, 1 - done)

        ep_step_count += 1
        global_step_count += 1

        if global_step_count > RANDOM_STEPS:
            if global_step_count % train_interval == 0:
                ddpg.learn()
                train_count += 1

        s = s_
        ep_reward += r

        if done or global_step_count == MAX_TOTAL_STEPS:
            break

    print('- Episode:', ep_num, ' Reward: %i' % int(ep_reward),
          # 'Explore: %.2f' % var,
          'Train count:', train_count,
          # 'Loss:', sum(ae_loss_his[-100:]) / 100.0, sum(p_loss_his[-100:]) / 100.0, sum(a_loss_his[-100:]) / 100.0
          )

    ep_num += 1

    if IS_SAVE_LOG:
        summary = sess.run(merged, feed_dict={var3: ep_reward})
        writer.add_summary(summary, global_step_count)


print('Running time: ', time.time() - t1)

