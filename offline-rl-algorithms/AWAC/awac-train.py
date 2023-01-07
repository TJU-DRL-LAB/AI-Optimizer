import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
import torch

def main(args):
    torch.set_num_threads(2)
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256, 256])
    optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=1e-4)

    awac = d3rlpy.algos.AWAC(actor_learning_rate=3e-4,
                             actor_encoder_factory=encoder,
                             actor_optim_factory=optim,
                             critic_learning_rate=3e-4,
                             batch_size=1024,
                             lam=1.0,
                             use_gpu=args.gpu)

    awac.fit(dataset.episodes,
             eval_episodes=test_episodes,
             n_steps=1000000,
             n_steps_per_epoch=1000,
             save_interval=10,
             tensorboard_dir='awac_runs/' + args.dataset + '/' + str(args.seed),
             scorers={
                 'environment': d3rlpy.metrics.evaluate_on_environment(env),
             },
             experiment_name=f"AWAC_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)