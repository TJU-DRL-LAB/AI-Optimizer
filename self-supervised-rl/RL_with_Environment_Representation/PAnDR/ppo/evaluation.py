import numpy as np
import torch
from ppo.envs import make_vec_envs

def evaluate(actor_critic, env_name, seed, num_processes, device, args):
    '''
    Evaluate the PPO agent. 
    '''
    eval_envs = make_vec_envs(args, 1, eval=True)

    eval_episode_rewards = []
    eval_episode_wins = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 100:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        for i in range(len(done)):
            if done[i]:
                eval_episode_rewards.append(reward[i].item())
                if reward[i].item() >= .93:
                    eval_episode_wins.append(1)
                else:
                    eval_episode_wins.append(0)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}, mean win {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.mean(eval_episode_wins)))
    return np.mean(eval_episode_rewards), np.mean(eval_episode_wins)
