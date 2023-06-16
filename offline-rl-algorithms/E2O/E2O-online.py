import argparse
import gym
import d3rlpy_new.d3rlpy
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # fix seed
    d3rlpy_new.d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    file_name = 'E2O-CQL-10_halfcheetah-medium-expert-v2_' + str(args.seed)
    file_path = "./d3rlpy_logs"
    filelist = os.listdir(file_path)
    for file in filelist:
        if file_name in file:
            file_name = file
    e2o = d3rlpy_new.d3rlpy.algos.E2O.from_json(
        'd3rlpy_logs/' + file_name + '/params.json',
        use_gpu=args.gpu
    )
    e2o.load_model(
        'd3rlpy_logs/' + file_name + '/model_1000000.pt'
    )

    buffer = d3rlpy_new.d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

    e2o.fit_online(env,
                   buffer,
                   eval_env=eval_env,
                   n_steps=250000,
                   n_steps_per_epoch=1000,
                   update_interval=1,
                   update_start_step=1000,
                   save_interval=1000000,
                   experiment_name=f"E2O-CQL-10_online_{args.env}-medium-expert_seed{args.seed}")


if __name__ == '__main__':
    main()
