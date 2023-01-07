import argparse
import gym
import d3rlpy
import torch


def main():

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    # load algorithm
    redq = d3rlpy.algos.REDQ(batch_size=256,
                                   actor_learning_rate=3e-4,
                                   critic_learning_rate=3e-4,
                                   temp_learning_rate=3e-4,
                                   use_gpu=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

    # start training
    redq.fit_online_redq(env,
                         buffer,
                         eval_env=eval_env,
                         n_steps=1000000,
                         n_steps_per_epoch=1000,
                         update_interval=1,
                         update_start_step=1000,
                         save_interval=100,
                         tensorboard_dir='runs',
                         experiment_name=f"REDQ_online_{args.env}_medium-replay_{args.seed}")


if __name__ == '__main__':
    main()
