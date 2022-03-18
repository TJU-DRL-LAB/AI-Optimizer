from cadm.dynamics.mlp_ensemble_cem_dynamics import MLPEnsembleCEMDynamicsModel
from cadm.trainers.mb_trainer import Trainer
from cadm.policies.mpc_controller import MPCController
from cadm.samplers.sampler import Sampler
from cadm.logger import logger
from cadm.envs.normalized_env import normalize
from cadm.utils.utils import ClassEncoder
from cadm.samplers.model_sample_processor import ModelSampleProcessor
from cadm.envs.config import get_environment_config

from tensorboardX import SummaryWriter
import json
import os
import gym
import argparse

def run_experiment(config):
    env, config = get_environment_config(config)

    # Save final config after editing config with respect to each environment.
    EXP_NAME = config['save_name']
    EXP_NAME += 'hidden_' + str(config['dim_hidden']) + '_lr_' + str(config['learning_rate'])
    EXP_NAME += '_horizon_' + str(config['horizon']) + '_seed_' + str(config['seed'])

    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last', only_test=config['only_test_flag'])
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    writer = SummaryWriter(exp_dir)

    dynamics_model = MLPEnsembleCEMDynamicsModel(
        name="dyn_model",
        env=env,
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['batch_size'],
        normalize_input=config['normalize_flag'],
        n_forwards=config['horizon'],
        n_candidates=config['n_candidates'],
        ensemble_size=config['ensemble_size'],
        n_particles=config['n_particles'],
        use_cem=config['use_cem'],
        deterministic=config['deterministic'],
        weight_decays=config['weight_decays'],
        weight_decay_coeff=config['weight_decay_coeff'],
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_rollouts=config['num_rollouts'],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
        random_flag=True,
        use_cem=config['use_cem'],
        horizon=config['horizon'],
    )

    sample_processor = ModelSampleProcessor(recurrent=False, writer=writer)

    algo = Trainer(
        env=env,
        env_flag=config['dataset'],
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
        num_test=config['num_test'],
        test_range=config['test_range'],
        total_test=config['total_test'],
        test_max_epochs=config['max_path_length'],
        no_test_flag=config['no_test_flag'],
        only_test_flag=config['only_test_flag'],
        use_cem=config['use_cem'],
        horizon=config['horizon'],
        writer=writer,
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------
    
    parser = argparse.ArgumentParser(description='conditional dynamics model')
    parser.add_argument('--save_name', default='NORMAL/', help="experiments name")
    parser.add_argument('--seed', type=int, default=0, help='random_seed')
    parser.add_argument('--dataset', default='halfcheetah', help='dataset flag')
    parser.add_argument('--hidden_size', type=int, default=200, help='size of hidden feature')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=5, help='training epochs per iteration')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--horizon', type=int, default=30, help='horrizon for planning')
    parser.add_argument('--normalize_flag', action='store_true', help='flag to normalize')
    parser.add_argument('--total_test', type=int, default=20, help='# of test')
    parser.add_argument('--n_candidate', type=int, default=200, help='candidate for planning')
    parser.add_argument('--no_test_flag', action='store_true', help='flag to disable test')
    parser.add_argument('--only_test_flag', action='store_true', help='flag to enable only test')
    parser.add_argument('--ensemble_size', type=int, default=5, help='size of ensembles')
    parser.add_argument('--n_particles', type=int, default=20, help='size of particles in trajectory sampling')
    parser.add_argument('--policy_type', type=str, default='CEM', help='Policy Type')
    parser.add_argument('--deterministic_flag', type=int, default=0, help='flag to use deterministic dynamics model')

    args = parser.parse_args()

    if args.normalize_flag:
        args.save_name = "/NORMALIZED/" + args.save_name
    else:
        args.save_name = "/RAW/" + args.save_name
    
    if args.dataset == 'cartpole':
        args.save_name = "/CARTPOLE/" + args.save_name
    elif args.dataset == 'pendulum':
        args.save_name = "/PENDULUM/" + args.save_name
    elif args.dataset == 'halfcheetah':
        args.save_name = "/HALFCHEETAH/" + args.save_name
    elif args.dataset == 'cripple_halfcheetah':
        args.save_name = "/CRIPPLE_HALFCHEETAH/" + args.save_name
    elif args.dataset == 'ant':
        args.save_name = "/ANT/" + args.save_name
    elif args.dataset == 'slim_humanoid':
        args.save_name = "/SLIM_HUMANOID/" + args.save_name
    else:
        raise ValueError(args.dataset)

    if args.deterministic_flag == 0:
        args.save_name += "PROB/"
    else:
        args.save_name += "DET/"

    if args.policy_type in ['RS', 'CEM']:
        args.save_name += "{}/".format(args.policy_type)
        args.save_name += "CAND_{}/".format(args.n_candidate)
    else:
        raise ValueError(args.policy_type)

    args.save_name += "BATCH_{}/".format(args.batch_size)
    args.save_name += "EPOCH_{}/".format(args.n_epochs)
    
    config = {
            # Policy
            'n_candidates': args.n_candidate,
            'horizon': args.horizon,

            # Policy - CEM Hyperparameters
            'use_cem': args.policy_type == 'CEM',

            # Environments
            'dataset': args.dataset,
            'normalize_flag': args.normalize_flag,
            'seed': args.seed,

            # Sampling
            'max_path_length': 200,
            'num_rollouts': 10,
            'n_parallel': 5,
            'initial_random_samples': True,

            # Training Hyperparameters
            'n_itr': 20,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'dynamic_model_epochs': args.n_epochs,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Testing Hyperparameters
            'total_test': args.total_test,
            'no_test_flag': args.no_test_flag,
            'only_test_flag': args.only_test_flag,

            # Dynamics Model Hyperparameters
            'dim_hidden': args.hidden_size,
            'hidden_sizes': (args.hidden_size,) * 4,
            'hidden_nonlinearity': 'swish',
            'deterministic': (args.deterministic_flag > 0),
            'weight_decays': (0.000025, 0.00005, 0.000075, 0.000075, 0.0001),
            'weight_decay_coeff': 1.0,

            # PE-TS Hyperparameters
            'ensemble_size': args.ensemble_size,
            'n_particles': args.n_particles,

            #  Other
            'save_name': args.save_name,
            'discount': 1.,
            }

    run_experiment(config)