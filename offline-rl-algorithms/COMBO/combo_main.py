# TH 20220131
import d4rl.gym_mujoco

import sys  
sys.path.append("/home/my/d3rlpy_baselines/")

import argparse
import d3rlpy
from d3rlpy.metrics import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics import dynamics_reward_prediction_error_scorer
from sklearn.model_selection import train_test_split

# TH 20220201
import static

from mopo_modTH import MOPOModTH

# added 20220727
from combo_modTH import COMBOModTH

import torch 
torch.set_num_threads(1)

def main(args):


    # create dataset without masks
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    # dataset, env, observations = d3rlpy.datasets.get_dataset(args.dataset)

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
                 # observations,
                 eval_episodes=test_episodes,
                 # n_steps=10000,
                 n_steps=100000,
                 scorers={
                     "obs_error": dynamics_observation_prediction_error_scorer,
                     "rew_error": dynamics_reward_prediction_error_scorer,
                 })

    if 'halfcheetah' in args.dataset:
        conservative_weight = 0.5
    elif 'medium-expert' in args.dataset:
        conservative_weight = 5.0
    elif 'random' in args.dataset or 'medium-replay' in args.dataset:
        if 'hopper' in args.dataset:
            conservative_weight = 1.0
        else:
            conservative_weight = 0.5
    elif 'medium' in args.dataset:
        conservative_weight = 5.0
    else:
        conservative_weight = 1.0

    if 'walker2d' in args.dataset:
        critic_learning_rate = 1e-4
        actor_learning_rate = 1e-5
        rollout_horizon=1
    else:
        critic_learning_rate = 3e-4
        actor_learning_rate = 1e-4
        rollout_horizon=5

    # prepare combo
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    termination_fn = static[args.dataset.split("-")[0]].termination_fn
    combo = COMBOModTH(dynamics=dynamics,
                               rollout_horizon=rollout_horizon,
                               actor_encoder_factory=encoder,
                               critic_encoder_factory=encoder,
                               actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               temp_learning_rate=actor_learning_rate,
                               conservative_weight=conservative_weight,
                               use_gpu=args.gpu,
                               n_critics=args.n_critics,
                               termination_fn = termination_fn,
                               )
    

    # train combo
    combo.fit(dataset.episodes,
              # observations=observations,
              eval_episodes=test_episodes,
            #   n_steps=2000,
              n_steps=1000000,
              n_steps_per_epoch=1000,
              save_interval=10,
              scorers={
                  "environment": d3rlpy.metrics.evaluate_on_environment(env),
                  'value_scale': d3rlpy.metrics.average_value_estimation_scorer
              },
              experiment_name=f"COMBO_{args.dataset}_{args.seed}")
    

    # # TH 20220203
    # termination_fn = static[args.dataset.split("-")[0]].termination_fn
    # mopo = MOPOModTH(dynamics=dynamics,
    #                          rollout_horizon=rollout_horizon,
    #                          lam=lam,
    #                          use_gpu=args.gpu,
    #                     termination_fn = termination_fn,
    #                     batch_size = 256,  # In MOPO original codebase, batch size is 256 over all envs. TH 20220203
    #                     entropy_target=-3.0 # In MOPO original codebase, entropy taget is -3 over all envs. TH
    #                     )

    # # train combo
    # mopo.fit(dataset.episodes,
    #          eval_episodes=test_episodes,
    #          n_steps=500000,
    #          n_steps_per_epoch=1000,
    #          save_interval=10,
    #          scorers={
    #              "environment": d3rlpy.metrics.evaluate_on_environment(env),
    #              'value_scale': d3rlpy.metrics.average_value_estimation_scorer
    #          },
    #          experiment_name=f"MOPO_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_critics', type=int, default=2)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args)