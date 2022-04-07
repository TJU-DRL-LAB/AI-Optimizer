import argparse
import tjuOfflineRL
from sklearn.model_selection import train_test_split


def main(args):
    dataset, env = tjuOfflineRL.datasets.get_dataset(args.dataset)

    # fix seed
    tjuOfflineRL.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    vae_encoder = tjuOfflineRL.models.encoders.VectorEncoderFactory([750, 750])

    if 'halfcheetah' in env.unwrapped.spec.id.lower():
        kernel = 'gaussian'
    else:
        kernel = 'laplacian'

    bear = tjuOfflineRL.algos.BEAR(actor_learning_rate=1e-4,
                             critic_learning_rate=3e-4,
                             imitator_learning_rate=3e-4,
                             alpha_learning_rate=1e-3,
                             imitator_encoder_factory=vae_encoder,
                             temp_learning_rate=0.0,
                             initial_temperature=1e-20,
                             batch_size=256,
                             mmd_sigma=20.0,
                             mmd_kernel=kernel,
                             n_mmd_action_samples=4,
                             alpha_threshold=0.05,
                             n_target_samples=10,
                             n_action_samples=100,
                             warmup_steps=40000,
                             use_gpu=args.gpu)

    bear.fit(dataset.episodes,
             eval_episodes=test_episodes,
             n_steps=1000000,
             n_steps_per_epoch=1000,
             save_interval=10,
             tensorboard_dir='runs/' + args.dataset + '/' + str(args.seed),
             scorers={
                 'environment': tjuOfflineRL.metrics.evaluate_on_environment(env),
             },
             experiment_name=f"BEAR_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)