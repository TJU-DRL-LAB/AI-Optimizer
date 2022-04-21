"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import random

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, RNN
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent, ExpPEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    env_eval = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params2']))
    tasks = env.get_all_task_idx()
    tasks_eval = env_eval.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder1 = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    use_next_state = variant['algo_params']['use_next_state']
    encoder_model = RNN if recurrent else MlpEncoder
    if use_next_state:
        context_encoder = encoder_model(
            hidden_sizes=[400, 400, 400],
            input_size=obs_dim * 2 + action_dim + reward_dim,
            output_size=context_encoder1,
        )
        context_encoder_target = encoder_model(
            hidden_sizes=[400, 400, 400],
            input_size=obs_dim + action_dim + reward_dim + obs_dim,
            output_size=context_encoder1,

        )
    else:
        context_encoder = encoder_model(
            hidden_sizes=[400, 400, 400],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=context_encoder1,
        )
        context_encoder_target = encoder_model(
            hidden_sizes=[400, 400, 400],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=context_encoder1,

        )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )#qnetwork1
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )#qnetwork2
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )#qnetwork3?
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )#actornetwork
    qf1_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )  # qnetwork1
    qf2_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )  # qnetwork2
    vf_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )  # qnetwork3?
    policy_exp = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )  # actornetwork
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        context_encoder_target,
        policy,
        **variant['algo_params']
    )
    exploration_agent = ExpPEARLAgent(
        latent_dim,
        context_encoder,
        context_encoder_target,
        policy_exp,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        env_eval=env_eval,
        train_tasks=list(tasks),
        eval_tasks=list(tasks_eval),
        nets=[agent, exploration_agent, qf1, qf2, vf, qf1_exp, qf2_exp, vf_exp],
        latent_dim=latent_dim,
        # to reduce pretrain time for debug
        # num_pretrain_steps_per_itr=100,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    # exp_id = 'debug' if DEBUG else None
    exp_id = variant['exp_id']
    my_base_log_dir = 'log'
    #  base_log_dir/exp_prefix/exp_id
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                      # base_log_dir=variant['util_params']['base_log_dir'],
                                      base_log_dir=my_base_log_dir,
                                      seed=variant['seed']
                                      )
    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=6)
@click.option('--exp_id', default='ccm')
def main(config, gpu, seed, exp_id):
    setup_seed(seed)
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['exp_id'] = exp_id
    variant['seed'] = seed
    experiment(variant)

if __name__ == "__main__":
    main()

