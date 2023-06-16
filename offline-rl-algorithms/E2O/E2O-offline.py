import argparse
import d3rlpy_new.d3rlpy
from sklearn.model_selection import train_test_split


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-expert-v2')
    parser.add_argument('--n_critic', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    dataset, env = d3rlpy_new.d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy_new.d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy_new.d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    e2o = d3rlpy_new.d3rlpy.algos.CQL(actor_learning_rate=3e-5,
                                      critic_learning_rate=3e-4,
                                      temp_learning_rate=1e-4,
                                      actor_encoder_factory=encoder,
                                      critic_encoder_factory=encoder,
                                      batch_size=256,
                                      n_action_samples=10,
                                      alpha_learning_rate=0.0,
                                      alpha_threshold=5.0,
                                      conservative_weight=5.0,
                                      n_critics=args.n_critic,
                                      use_gpu=args.gpu)

    e2o.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            save_interval=1000,
            scorers={
                'environment': d3rlpy_new.d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy_new.d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"E2O-CQL-{args.n_critic}_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()
