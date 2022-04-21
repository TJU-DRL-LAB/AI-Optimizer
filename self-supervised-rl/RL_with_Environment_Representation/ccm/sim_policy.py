
import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg
    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    env = CameraWrapper(NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params'])), variant['util_params']['gpu_id'])
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[150, 150],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

    # loop through tasks collecting rollouts
    all_rets = []
    video_frames = []
    for idx in eval_tasks:
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
            paths.append(path)
            if save_video:
                video_frames += [t['frame'] for t in path['env_infos']]
            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)
        all_rets.append([sum(p['rewards']) for p in paths])

    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        print('trajectory {}, avg return: {} \n'.format(i, ret))


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=3)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=True)
def main(config, path, num_trajs, deterministic, video):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video)


if __name__ == "__main__":
    main()

