# TH 20220131
import d4rl.gym_mujoco

import argparse
import d3rlpy
from d3rlpy.metrics import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics import dynamics_reward_prediction_error_scorer
from sklearn.model_selection import train_test_split

# TH 20220201
import static
from mopo_modTH import MOPOModTH

PARAMETER_TABLE = {
    'halfcheetah-random-v0': (5, 0.5),
    'hopper-random-v0': (5, 1),
    'walker2d-random-v0': (1, 1),
    'halfcheetah-medium-v0': (1, 1),
    'hopper-medium-v0': (5, 5),
    'walker2d-medium-v0': (5, 5),
    'halfcheetah-medium-replay-v0': (5, 1),
    'hopper-medium-replay-v0': (5, 1),
    'walker2d-medium-replay-v0': (1, 1),
    'halfcheetah-medium-expert-v0': (5, 1),
    'hopper-medium-expert-v0': (5, 1),
    'walker2d-medium-expert-v0': (1, 2)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    # create dataset without masks
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    # prepare dynamics model
    dynamics_encoder = d3rlpy.models.encoders.VectorEncoderFactory(
        hidden_units=[200, 200, 200, 200],
        activation='swish',
    )
    dynamics_optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=2.5e-5)
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
        encoder_factory=dynamics_encoder,
        optim_factory=dynamics_optim,
        learning_rate=1e-3,
        n_ensembles=5,
        use_gpu=args.gpu,
    )

    # train dynamics model
    dynamics.fit(dataset.episodes,
                 eval_episodes=test_episodes,
                 n_steps=100000,
                 scorers={
                     "obs_error": dynamics_observation_prediction_error_scorer,
                     "rew_error": dynamics_reward_prediction_error_scorer,
                 })

    if args.dataset in PARAMETER_TABLE:
        rollout_horizon, lam = PARAMETER_TABLE[args.dataset]
    else:
        rollout_horizon, lam = 5, 1

    # # TH 20220203
    termination_fn = static[args.dataset.split("-")[0]].termination_fn
    mopo = MOPOModTH(dynamics=dynamics,
                             rollout_horizon=rollout_horizon,
                             lam=lam,
                             use_gpu=args.gpu,
                        termination_fn = termination_fn,
                        batch_size = 256,  # In MOPO original codebase, batch size is 256 over all envs. TH 20220203
                        entropy_target=-3.0 # In MOPO original codebase, entropy taget is -3 over all envs. TH
                        )

    # train combo
    mopo.fit(dataset.episodes,
             eval_episodes=test_episodes,
             n_steps=500000,
             n_steps_per_epoch=1000,
             save_interval=10,
             scorers={
                 "environment": d3rlpy.metrics.evaluate_on_environment(env),
                 'value_scale': d3rlpy.metrics.average_value_estimation_scorer
             },
             experiment_name=f"MOPO_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()