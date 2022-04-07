import argparse
import tjuOfflineRL
from sklearn.model_selection import train_test_split
import torch

def main(args):
    torch.set_num_threads(2)
    dataset, env = tjuOfflineRL.datasets.get_dataset(args.dataset)

    # fix seed
    tjuOfflineRL.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    vae_encoder = tjuOfflineRL.models.encoders.VectorEncoderFactory([750, 750])
    rl_encoder = tjuOfflineRL.models.encoders.VectorEncoderFactory([400, 300])

    bcq = tjuOfflineRL.algos.BCQ(actor_encoder_factory=rl_encoder,
                           actor_learning_rate=1e-3,
                           critic_encoder_factory=rl_encoder,
                           critic_learning_rate=1e-3,
                           imitator_encoder_factory=vae_encoder,
                           imitator_learning_rate=1e-3,
                           batch_size=256,
                           lam=0.75,
                           action_flexibility=0.05,
                           n_action_samples=100,
                           use_gpu=args.gpu)

    bcq.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            save_interval=10,
            tensorboard_dir='bcq_runs/' + args.dataset + '/' + str(args.seed),
            scorers={
                'environment': tjuOfflineRL.metrics.evaluate_on_environment(env),
            },
            experiment_name=f"BCQ_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)