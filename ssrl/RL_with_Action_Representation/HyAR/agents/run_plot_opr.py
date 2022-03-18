from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=777, help='random seed')

parser.add_argument("--pr-type", type=str, default='e2e_opr', help='policy representation type')
# parser.add_argument("--pr-type", type=str, default='cl_opr', help='policy representation type')
# parser.add_argument("--pr-type", type=str, default='cl_opr_fix2', help='policy representation type')
# parser.add_argument("--pr-type", type=str, default='aux_opr', help='policy representation type')

parser.add_argument("--visual-type", type=int, default=1, help='visual type')
# parser.add_argument("--visual-type", type=int, default=2, help='visual type')
parser.add_argument("--is-plot-marker", type=bool, default=True, help='is plot marker')
# parser.add_argument("--is-plot-marker", type=bool, default=False, help='is plot marker')
parser.add_argument("--is-track-learning", type=bool, default=False, help='track learning process')
# parser.add_argument("--is-track-learning", type=bool, default=True, help='track learning process')
parser.add_argument("--is-save-tsne-single", type=bool, default=True, help='is save tsne single')


args = parser.parse_args()
pr_type = args.pr_type

np.random.seed(args.seed)
tf.set_random_seed(args.seed)


# FIXME 0104 - load policy data
env_name = 'HalfCheetah-v1'
# env_name = 'Ant-v1'
# env_name = 'Hopper-v1'
# env_name = 'Walker2d-v1'


# ------------------------- load model -------------------------------
data_path = './../run_for_ckpts/ckpts/'
data_path += pr_type + '_0.0001/' + env_name + '/'
# data_path += 'seed' + str(args.seed) + '/'


policy_data_list = []
policy_eval_list = []
policy_track_list = []
pr_list_for_plot = []

ckpt_num = 20 if env_name == 'Ant-v1' else 10
model_idx = [0] + [10 * i + 9 for i in range(10)]

# -----------------------------
print('- Loading data...')
policy_num_list = []
for data_seed in range(6):
    # FIXME 0825 - for analysis
    # if data_seed not in [1,2,3,4,5]:
    #     continue
    policy_data_path = data_path + 'seed' + str(data_seed) + '/'
    policy_data_path += env_name + '_ppo_pevf_' + pr_type + '_s' + str(data_seed) + '.npz'

    npz_data = np.load(policy_data_path)
    policy_data = npz_data['pr_data_4plot']
    eval_data = npz_data['avg_return_4plot']

    policy_data_list.append(policy_data)
    policy_eval_list.append(eval_data)
    policy_track_list.append(np.array(range(policy_data.shape[0])))

    policy_num_list.append(policy_data.shape[0])

    print('-- Loading:', policy_data_path)
    print('-- Max policy return:', np.max(eval_data), ', Policy num:', policy_data.shape[0])

policy_data_2plot = np.concatenate(policy_data_list, axis=0)
policy_eval_2plot = np.concatenate(policy_eval_list, axis=0)
policy_track_2plot = np.concatenate(policy_track_list, axis=0)

# -----------------------------
# if
import gym

sess = tf.Session()
env = gym.make(env_name)
env = env.unwrapped
s_dim = env.observation_space.shape[0] + 1
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high


if pr_type == 'e2e_opr':
    from networks.for_ckpts.ppo_pevf_e2e_opr1_ckpt import PPO_PEVF
elif pr_type == 'cl_opr':
    from networks.for_ckpts.ppo_pevf_cl_opr_ckpt import PPO_PEVF
elif pr_type == 'cl_opr_fix2':
    from networks.for_ckpts.ppo_pevf_cl_opr_fix2_ckpt import PPO_PEVF
elif pr_type == 'aux_opr':
    from networks.for_ckpts.ppo_pevf_aux_opr_ckpt import PPO_PEVF
else:
    raise NotImplementedError

pr_model = PPO_PEVF(s_dim=s_dim, a_dim=a_dim, pr_dim=64, sess=sess,
                    gamma=0.99, k=64, prev_type=0, batch_size=128,
                    lr_c=0.001, lr_a=0.0001, lr_pc=0.001,
                    # FIXME 0522
                    memory_size=200000, policy_size=10000,
                    # memory_size=1000000, policy_size=100000,
                    pr_model=None)

init_pr_data = pr_model.get_params()
print(init_pr_data.tolist())
# -------------------------
# FIXME 0825 - use a single model to infer
# model_seed = 5
# model_path = data_path + 'seed' + str(model_seed) + '/'
# model_path += env_name + '_' + pr_type + '_s' + str(model_seed) + '.ckpt'
# pr_model.load_checkpoint(load_path=model_path + '-' + str(99))
# pr = sess.run(pr_model.surface_pr, feed_dict={pr_model.param_ph: policy_data_2plot})

pr_list_2avg = []
ckpt_num = 2
model_num = 6

if args.visual_type == 0:
    # FIXME 0825 - use multiple models to infer: avg over seeds
    for model_seed in range(model_num):
        inner_list = []
        for ckpt_idx in range(ckpt_num):
            model_path = data_path + 'seed' + str(model_seed) + '/'
            model_path += env_name + '_' + pr_type + '_s' + str(model_seed) + '.ckpt'
            pr_model.load_checkpoint(load_path=model_path + '-' + str(99 - 20 * ckpt_idx))

            if pr_type == 'e2e_opr':
                tmp_pr = sess.run(pr_model.surface_pr, feed_dict={pr_model.param_ph: policy_data_2plot})
            else:
                tmp_pr = sess.run(pr_model.original_pr, feed_dict={pr_model.param_ph: policy_data_2plot})
            inner_list.append(tmp_pr)

        inner_concat = np.concatenate(inner_list, axis=1)
        pr_list_2avg.append(inner_concat[np.newaxis,])

elif args.visual_type == 1:
    # FIXME 0825 - use multiple models to infer: avg over recent ckpts
    for ckpt_idx in range(ckpt_num):
        inner_list = []
        for model_seed in range(model_num):
            model_path = data_path + 'seed' + str(model_seed) + '/'
            model_path += env_name + '_' + pr_type + '_s' + str(model_seed) + '.ckpt'
            if env_name == 'Ant-v1':
                pr_model.load_checkpoint(load_path=model_path + '-' + str(199 - 20 * ckpt_idx))
            else:
                pr_model.load_checkpoint(load_path=model_path + '-' + str(99 - 20 * ckpt_idx))

            if pr_type == 'e2e_opr':
                tmp_pr = sess.run(pr_model.surface_pr, feed_dict={pr_model.param_ph: policy_data_2plot})
            elif pr_type == 'cl_opr_fix2':
                tmp_pr = sess.run(pr_model.original_pr, feed_dict={pr_model.param_ph: policy_data_2plot,
                                                                   pr_model.fc1_sample_num_ph: [pr_model.k],
                                                                   pr_model.fc2_sample_num_ph: [pr_model.k],
                                                                   })
            else:
                tmp_pr = sess.run(pr_model.original_pr, feed_dict={pr_model.param_ph: policy_data_2plot,
                                                                   })
            inner_list.append(tmp_pr)

        inner_concat = np.concatenate(inner_list, axis=1)
        pr_list_2avg.append(inner_concat[np.newaxis,])

elif args.visual_type == 2:
    for ckpt_idx in range(ckpt_num):
        inner_list = []
        model_seed = model_num

        model_path = data_path + 'seed' + str(model_seed) + '/'
        model_path += env_name + '_' + pr_type + '_s' + str(model_seed) + '.ckpt'
        if env_name == 'Ant-v1':
            pr_model.load_checkpoint(load_path=model_path + '-' + str(199 - 20 * ckpt_idx))
        else:
            pr_model.load_checkpoint(load_path=model_path + '-' + str(99 - 20 * ckpt_idx))

        if pr_type == 'e2e_opr':
            tmp_pr = sess.run(pr_model.surface_pr, feed_dict={pr_model.param_ph: policy_data_2plot})
        elif pr_type == 'cl_opr_fix2':
            tmp_pr = sess.run(pr_model.original_pr, feed_dict={pr_model.param_ph: policy_data_2plot,
                                                               pr_model.fc1_sample_num_ph: [pr_model.k],
                                                               pr_model.fc2_sample_num_ph: [pr_model.k],
                                                               })
        else:
            tmp_pr = sess.run(pr_model.original_pr, feed_dict={pr_model.param_ph: policy_data_2plot,
                                                               })
        inner_list.append(tmp_pr)

        inner_concat = np.concatenate(inner_list, axis=1)
        pr_list_2avg.append(inner_concat[np.newaxis,])
else:
    raise NotImplementedError

pr_concat = np.concatenate(pr_list_2avg, axis=0)
pr = np.mean(pr_concat, axis=0)

# ------------------------- visual analysis -------------------------------
plot_num = 6
plot_policy_num = sum(policy_num_list[:plot_num])
x = pr[:plot_policy_num,]
if args.is_track_learning:
    c = policy_track_2plot[:plot_policy_num,]
else:
    c = policy_eval_2plot[:plot_policy_num,]

tSNE = TSNE(n_components=2,
            # perplexity=5,
            perplexity=2,
            early_exaggeration=6.0, learning_rate=500, n_iter=800,
            # init='random',
            init='pca',
            random_state=111)
# tSNE = TSNE(n_components=2, perplexity=30, n_iter=500, learning_rate=300, random_state=222, init='pca')
pca = PCA(n_components=2)

X_tsne = tSNE.fit_transform(x)
X_pca = pca.fit_transform(x)


# ------------------------ plot -------------------------------
fig = plt.figure(figsize=(12, 6))
size = 16 if env_name == 'Ant-v1' else 30
if args.is_plot_marker:
    # ------------------------ data pre -----------------------------
    # FIXME 0825 - add extra min and max to make cmap consistant
    c_appendix = [np.max(c), np.min(c)]

    c_max_xt = X_tsne[np.argmax(c)][np.newaxis,]
    c_min_xt = X_tsne[np.argmin(c)][np.newaxis,]
    xt_appendix = np.concatenate([c_max_xt, c_min_xt], axis=0)

    c_max_xp = X_pca[np.argmax(c)][np.newaxis,]
    c_min_xp = X_pca[np.argmin(c)][np.newaxis,]
    xp_appendix = np.concatenate([c_max_xp, c_min_xp], axis=0)

    marker_list = ['*', '.', 'v', '1', '+', 'x', 'd', 'h']


    plt.subplot(121)
    for pid in range(plot_num):
        start_idx = 0 if pid == 0 else sum(policy_num_list[:pid])
        end_idx = sum(policy_num_list[:pid + 1])
        plot_x = np.concatenate([X_tsne[start_idx:end_idx], xt_appendix], axis=0)
        plot_c = np.concatenate([c[start_idx:end_idx], c_appendix], axis=0)
        plt.scatter(plot_x[:,0], plot_x[:,1], c=plot_c, label='Trial-' + str(pid),
                    alpha=0.7, cmap=plt.cm.get_cmap('rainbow', 50), marker=marker_list[pid],
                    s=size)
    plt.legend()
    plt.title('tSNE')

    plt.subplot(122)
    for pid in range(plot_num):
        start_idx = 0 if pid == 0 else sum(policy_num_list[:pid])
        end_idx = sum(policy_num_list[:pid + 1])
        plot_x = np.concatenate([X_pca[start_idx:end_idx], xp_appendix], axis=0)
        plot_c = np.concatenate([c[start_idx:end_idx], c_appendix], axis=0)
        plt.scatter(plot_x[:, 0], plot_x[:, 1], c=plot_c, label='Trial-' + str(pid),
                    alpha=0.7, cmap=plt.cm.get_cmap('rainbow', 50), marker=marker_list[pid],
                    s=size)
    plt.legend()
    plt.title('PCA')

else:
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c, label='t-SNE', alpha=0.7, cmap=plt.cm.get_cmap('rainbow', 50), s=size)
    plt.legend()

    plt.subplot(122)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=c, label='PCA', alpha=0.7, cmap=plt.cm.get_cmap('rainbow', 50), s=size)
    plt.legend()

plt.colorbar()

# if args.is_track_learning:
#     plt.savefig('tSNE_PCA_' + env_name + '_' + pr_type + '_vt' + str(args.visual_type) + '_learning.pdf')
# else:
#     plt.savefig('tSNE_PCA_' + env_name + '_' + pr_type + '_vt' + str(args.visual_type) + '_performance.pdf')

plt.show()

print()

# if args.is_save_tsne_single:
#     fig = plt.figure(figsize=(12, 6))



