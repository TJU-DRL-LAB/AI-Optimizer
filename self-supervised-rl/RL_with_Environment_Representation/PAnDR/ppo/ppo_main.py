import os
from collections import deque
import sys
sys.path.append('/home/st/policy-dynamics-value-functions-master/')
import gym
import numpy as np
import torch

import algo 
import utils 
from arguments import get_args
from envs import make_vec_envs 
from model import Policy 
from storage import RolloutStorage
from evaluation import evaluate

import myant
import myswimmer 
import myspaceship

def main():
    '''
    Train PPO policies on each of the training environments.
    '''
    args = get_args()
    
    try:
        os.makedirs(args.log_dir)
    except OSError:
        pass

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cpu")# "cuda:0" if args.cuda else 

    envs = make_vec_envs(args, device)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    ep_reward = np.zeros(args.num_processes)
    episode_rewards = deque(maxlen=100)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        # decrease learning rate linearly
        utils.update_linear_schedule(
            agent.optimizer, j, num_updates, args.lr
        )

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obs reward and next obs
            obs, reward, done, infos = envs.step(action)
            if 'spaceship' in args.env_name: # spaceship, swimmer
                for i in range(len(done)):
                    if done[i]:
                        episode_rewards.append(reward[i].item())
            # elif 'swimmer' in args.env_name:
            else:
                for i in range(len(done)):
                    ep_reward[i] += reward[i].numpy().item()        
                    if done[i]:
                        episode_rewards.append(ep_reward[i])
                        ep_reward[i] = 0  
            # if 'ant' in args.env_name:
            #     for info in infos:
            #         if 'episode' in info.keys():
            #             episode_rewards.append(info['episode']['r'])
                        
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, True, args.gamma,
                                 args.gae_lambda, True)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass


            torch.save(
                actor_critic.state_dict(), 
                os.path.join(args.save_dir, "ppo.{}.env{}.seed{}.pt"\
                    .format(args.env_name, args.default_ind, args.seed))
            )

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "\nUpdates {}, num timesteps {}, Last {} training episodes: \
                \n mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}"
                .format(j, total_num_steps, 
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, device)

    envs.close()

if __name__ == "__main__":
    main()