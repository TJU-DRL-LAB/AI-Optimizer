import gym
import myspaceship
import torch
import torch.nn.functional as F


MAX_EPISODE_STEPS_ANT = 256
MAX_EPISODE_STEPS_SWIMMER = 1000 
MAX_EPISODE_STEPS_SPACESHIP = 50 


def make_one_hot(x, nb_digits=3, batch_size=1):
    '''
    Convert int to one hot tensor
    '''
    y = x.reshape(batch_size, 1)

    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.long(), 1)

    return y_onehot


class EnvSamplerEmb():
    '''
    Environment sampler object for training the 
    policy and dynamics embeddings.
    '''
    def __init__(self, env, base_policy, args):
        if args.env_name.startswith('spaceship'):
            self.env = env.venv.envs[0]
        elif args.env_name.startswith('myswimmer'):
            self.env = env.venv.envs[0].env.env
        else:
            self.env = env.venv.envs[0].env.env.env
        self.env.reset()
        self.env1 = env
        self.base_policy = base_policy
        self.args = args
        self.env_name = args.env_name
        self.action_dim = self.env1.action_space.shape[0]
        self.state_dim = self.env1.observation_space.shape[0]
        self.enc_input_size = 2*self.state_dim + self.action_dim
        self.inf_num_steps = args.inf_num_steps        
        self.env_max_seq_length = args.inf_num_steps * self.enc_input_size
        self.max_seq_length = args.max_num_steps * self.enc_input_size 

        if 'ant' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_ANT
        elif 'swimmer' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_SWIMMER
        elif 'spaceship' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_SPACESHIP

    def reset(self):
        '''
        Reset the environment to an environment that might have different dynamics.
        '''
        self.env1.reset()

    def reset_same(self):
        '''
        Reset the environment to an environment with identical dynamics.
        '''
        return self.env.reset(same=True)

    def sample_env_context(self, policy_fn, env_idx=None):
        '''
        Generate a few steps in the new environment. 
        Use these transitions to infer the environment's dynamics.  
        '''
        src = []
        this_src_batch = []
        if env_idx is not None:
            state = self.env.reset(env_id=env_idx)
        else:
            state = self.reset_same()
        state = torch.tensor(state).float()
        done = False

        max_num_steps = self.inf_num_steps
        
        for t in range(max_num_steps):
            recurrent_hidden_state = torch.zeros(
                1, policy_fn.recurrent_hidden_state_size, device=self.args.device)
            mask = torch.zeros(1, 1, device=self.args.device)
            policy_fn = policy_fn.float()
            action = policy_fn.act(
                        state.squeeze().unsqueeze(0).to(torch.device(self.args.device)).float(),
                        recurrent_hidden_state.float(), mask.float(),
                        deterministic=True)[1].detach()

            action_tensor = action.float().reshape(self.action_dim)
            state_action_tensor = torch.cat((state.squeeze().to(device=self.args.device),
                action_tensor.reshape(self.action_dim)), dim=0)
            if not done:
                next_state, _, done, _ = self.env.step(*action.cpu().numpy())
                next_state_tensor = torch.FloatTensor(next_state)
                state = next_state_tensor
                state_action_state_tensor = torch.cat((state_action_tensor.to(device=self.args.device), \
                    next_state_tensor.to(device=self.args.device)), dim=0).unsqueeze(1)
            else:
                state_action_state_tensor = torch.FloatTensor([0 for _ in range(self.enc_input_size)])\
                    .reshape(self.enc_input_size, 1)
            src.append(state_action_state_tensor)
            src_pad = F.pad(torch.stack(src).reshape((t + 1) * self.enc_input_size), \
                        (0, self.max_seq_length - (t + 1) * len(state_action_state_tensor)))

            this_src_batch.append(src_pad)

        return this_src_batch[-1]

    def sample_policy(self, policy_idx=0):
        '''
        Sample a policy from your set of pretrained policies. 
        '''

        print("---------",len(self.base_policy))
        return self.base_policy[policy_idx]

    def generic_step(self, policy_fn, state):
        '''
        Take a step in the environment 
        '''
        action = policy_fn(state.squeeze().unsqueeze(0))
        action_tensor = torch.FloatTensor(action) 
        state_action_tensor = torch.cat((state.squeeze(),
                                            action_tensor.squeeze()), dim=0)

        next_state, reward, done, _ = self.env.step(action.numpy())
        next_state_tensor = torch.Tensor(next_state).float()
        if not done:
            sas_tensor = torch.cat((state_action_tensor, next_state_tensor), dim=0).unsqueeze(1)
        else:
            sas_tensor = torch.FloatTensor([0 for _ in range(self.enc_input_size)]) \
                .reshape(self.enc_input_size, 1)

        res = {'state': next_state_tensor, 'action': action_tensor, 'sa_tensor': state_action_tensor,
               'sas_tensor': sas_tensor, 'reward': reward, 'done': done}
        return res

    def get_decoded_traj(self, args, init_state, init_obs, policy_emb, decoder, env_idx=0, verbose=False):
        '''
        Decode a trajectory using the policy decoder conditioned on a given policy embedding, 
        in a given environment for a given initial state. Works with the Spaceship environment.
        '''
        state = self.env.reset(env_id=env_idx)
        state = torch.FloatTensor(state)

        device = args.device
        done = False
        episode_reward = 0

        all_emb_state = []
        all_recurrent_state = []
        all_mask = []
        all_action = []
        for t in range(args.max_num_steps):
            recurrent_hidden_state = torch.zeros(policy_emb.shape[0],
                decoder.recurrent_hidden_state_size, device=device, requires_grad=True).float()
            mask_dec = torch.zeros(policy_emb.shape[0], 1, device=device, requires_grad=True).float()
            emb_state_input = torch.cat((policy_emb.to(device), state.unsqueeze(0).to(device)), dim=1).to(device)
            
            action = decoder.act(emb_state_input, recurrent_hidden_state, mask_dec,
                deterministic=True)[1]

            action_flat = action.squeeze().cpu().detach().numpy()
            action = action.cpu().detach().numpy()

            all_emb_state.append(emb_state_input)
            all_recurrent_state.append(recurrent_hidden_state)
            all_mask.append(mask_dec)
            all_action.append(torch.tensor(action).to(device))

            next_state, reward, done, _ = self.env.step(action_flat)
            episode_reward = args.gamma * episode_reward + reward
            state = torch.FloatTensor(next_state)
            if done: 
                break

        return all_emb_state, all_recurrent_state, all_mask, all_action

    def get_decoded_traj_mujoco(self, args, init_state, init_obs, policy_emb, decoder, env_idx=0, verbose=False):
        '''
        Decode a trajectory using the policy decoder conditioned on a given policy embedding, 
        in a given environment for a given initial state. Works with MuJoCo environments (Ant and Swimmer).
        '''
        self.env.reset(env_id=env_idx)
        self.env.sim.set_state(init_state)
        state = init_obs 

        device = args.device
        done = False
        episode_reward = 0
        t = 0

        all_emb_state = []
        all_recurrent_state = []
        all_mask = []
        all_action = []
        for t in range(self.max_episode_steps):
            t += 1
            recurrent_hidden_state = torch.zeros(policy_emb.shape[0],
                decoder.recurrent_hidden_state_size, device=device, requires_grad=True).float()
            mask_dec = torch.zeros(policy_emb.shape[0], 1, device=device, requires_grad=True).float()
            emb_state_input = torch.cat((policy_emb.to(device), state.unsqueeze(0).to(device)), dim=1).to(device)
            action = decoder.act(emb_state_input, recurrent_hidden_state, mask_dec,
                deterministic=True)[1]

            action_flat = action.squeeze().cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            
            all_emb_state.append(emb_state_input)
            all_recurrent_state.append(recurrent_hidden_state)
            all_mask.append(mask_dec)
            all_action.append(torch.tensor(action).to(device))

            next_state, reward, done, _ = self.env.step(action_flat)
            episode_reward = args.gamma * episode_reward + reward
            state = torch.FloatTensor(next_state)
            if done: 
                break

        if args.norm_reward:
            episode_reward = (episode_reward - args.min_reward) / (args.max_reward - args.min_reward)
        
        return all_emb_state, all_recurrent_state, all_mask, all_action

    def sample_data_ds(self, policy_idx=0, env_idx=None):
        '''
        Sample data using a given policy.
        '''
        done = False
        state_action_batch = []
        state_action_state_batch = []
        tgt_batch = []
        mask_batch = []
        src = []
        masks = []

        if env_idx is not None:
            init_state = self.env.reset(env_id=env_idx)
        else:
            init_state = self.env.reset()

        trajectory = self.sample_policy(policy_idx=policy_idx)
        state_tensor = torch.tensor(init_state).unsqueeze(0)
        recurrent_hidden_state = torch.zeros(
            1, trajectory.recurrent_hidden_state_size, device=self.args.device)
        mask = torch.zeros(1, 1, device=self.args.device)
        trajectory = trajectory.float().to(self.args.device)
        action = trajectory.act(
            state_tensor.to(torch.device(self.args.device)).float(),
            recurrent_hidden_state.float(), mask.float(),
            deterministic=True)[1]
        action_tensor = action.float().reshape(self.action_dim)
        state_action_tensor = torch.cat([
            state_tensor.to(torch.device(self.args.device)).float().squeeze(),
            action_tensor], dim=0)

        for t in range(self.args.max_num_steps):
            state_action_batch.append(state_action_tensor.unsqueeze(1))

            masks.append(torch.FloatTensor([done == False]))
            mask_batch.append(torch.FloatTensor([done == False]))

            # 这里没有check done=true咋整　不过max num step=50可能比较小不会到done??
            state_tensor, _, done, _ = self.env.step(action_tensor.cpu().detach().numpy())
            next_state_tensor = torch.Tensor(state_tensor).float().to(self.args.device)
            # print(state_action_tensor,'---------',next_state_tensor)
            state_action_state_tensor = torch.cat((state_action_tensor, next_state_tensor), dim=0).unsqueeze(1)
            state_action_state_batch.append(state_action_state_tensor)
            tgt_batch.append(action_tensor.detach())  # action是target

            state_tensor = torch.tensor(state_tensor).unsqueeze(0)
            recurrent_hidden_state = torch.zeros(
                1, trajectory.recurrent_hidden_state_size, device=self.args.device)
            mask = torch.zeros(1, 1, device=self.args.device)
            trajectory = trajectory.float().to(self.args.device)
            action = trajectory.act(
                state_tensor.to(torch.device(self.args.device)).float(),
                recurrent_hidden_state.float(), mask.float(),
                deterministic=True)[1].detach()
            action_tensor = action.float().reshape(self.action_dim)
            state_action_tensor = torch.cat([
                state_tensor.to(torch.device(self.args.device)).float().squeeze(),
                action_tensor], dim=0)
        # print(len(src), src[0].shape)

        return state_action_batch, mask_batch, state_action_state_batch# , tgt_batch, src_batch, mask_batch, mask_batch_all

    def sample_policy_data_ds(self, policy_idx=0, env_idx=None):
        '''
        Sample data using a given policy.
        '''
        done = False
        state_action_batch = []
        tgt_batch = []
        src_batch = []
        state_batch = []
        mask_batch = []
        mask_batch_all = []
        src = []
        masks = []

        if env_idx is not None:
            init_state = self.env.reset(env_id=env_idx)
        else:
            init_state = self.env.reset()

        trajectory = self.sample_policy(policy_idx=policy_idx)
        state_tensor = torch.tensor(init_state).unsqueeze(0)
        recurrent_hidden_state = torch.zeros(
            1, trajectory.recurrent_hidden_state_size, device=self.args.device)
        mask = torch.zeros(1, 1, device=self.args.device)
        trajectory = trajectory.float().to(self.args.device)
        action = trajectory.act(
            state_tensor.to(torch.device(self.args.device)).float(),
            recurrent_hidden_state.float(), mask.float(),
            deterministic=True)[1]
        action_tensor = action.float().reshape(self.action_dim)
        state_action_tensor = torch.cat([
            state_tensor.to(torch.device(self.args.device)).float().squeeze(),
            action_tensor], dim=0).unsqueeze(1)

        for t in range(self.args.max_num_steps):
            state_action_batch.append(state_action_tensor)
            state_batch.append(state_tensor)
            src.append(state_action_tensor)
            masks.append(torch.FloatTensor([done == False]))
            mask_batch.append(torch.FloatTensor([done == False]))

            # 这里没有check done=true咋整　不过max num step=50可能比较小不会到done??
            state_tensor, _, done, _ = self.env.step(action_tensor.cpu().detach().numpy())

            tgt_batch.append(action_tensor.detach())  # action是target

            state_tensor = torch.tensor(state_tensor).unsqueeze(0)
            recurrent_hidden_state = torch.zeros(
                1, trajectory.recurrent_hidden_state_size, device=self.args.device)
            mask = torch.zeros(1, 1, device=self.args.device)
            trajectory = trajectory.float().to(self.args.device)
            action = trajectory.act(
                state_tensor.to(torch.device(self.args.device)).float(),
                recurrent_hidden_state.float(), mask.float(),
                deterministic=True)[1].detach()
            action_tensor = action.float().reshape(self.action_dim)
            state_action_tensor = torch.cat([
                state_tensor.to(torch.device(self.args.device)).float().squeeze(),
                action_tensor], dim=0).unsqueeze(1)
        # print(len(src), src[0].shape)

        return state_batch, tgt_batch, src_batch, mask_batch, mask_batch_all, state_action_batch

    def sample_policy_data(self, policy_idx=0, env_idx=None):
        '''
        Sample data using a given policy. 
        '''
        done = False
        state_batch = []
        tgt_batch = []
        src_batch = []
        mask_batch = []
        mask_batch_all = []
        src = []
        masks = []

        if env_idx is not None:
            init_state = self.env.reset(env_id=env_idx)
        else:
            init_state = self.env.reset()

        trajectory = self.sample_policy(policy_idx=policy_idx)
        state_tensor = torch.tensor(init_state).unsqueeze(0) 
        recurrent_hidden_state = torch.zeros(
            1, trajectory.recurrent_hidden_state_size, device=self.args.device)
        mask = torch.zeros(1, 1, device=self.args.device)
        trajectory = trajectory.float().to(self.args.device)
        action = trajectory.act(
            state_tensor.to(torch.device(self.args.device)).float(),
            recurrent_hidden_state.float(), mask.float(),
            deterministic=True)[1]
        action_tensor = action.float().reshape(self.action_dim)
        state_action_tensor = torch.cat([
            state_tensor.to(torch.device(self.args.device)).float().squeeze(),
            action_tensor], dim=0).unsqueeze(1)

        for t in range(self.args.max_num_steps):
            state_batch.append(state_tensor)
            src.append(state_action_tensor)
            masks.append(torch.FloatTensor([done == False]))
            mask_batch.append(torch.FloatTensor([done == False]))

            # 这里没有check done=true咋整　不过max num step=50可能比较小不会到done??
            state_tensor, _, done, _ = self.env.step(action_tensor.cpu().detach().numpy())
            
            tgt_batch.append(action_tensor.detach())    # action是target
            
            state_tensor = torch.tensor(state_tensor).unsqueeze(0)
            recurrent_hidden_state = torch.zeros(
                1, trajectory.recurrent_hidden_state_size, device=self.args.device)
            mask = torch.zeros(1, 1, device=self.args.device)
            trajectory = trajectory.float().to(self.args.device)
            action = trajectory.act(
                state_tensor.to(torch.device(self.args.device)).float(),
                recurrent_hidden_state.float(), mask.float(),
                deterministic=True)[1].detach()
            action_tensor = action.float().reshape(self.action_dim)
            state_action_tensor = torch.cat([
                state_tensor.to(torch.device(self.args.device)).float().squeeze(),
                action_tensor], dim=0).unsqueeze(1)
        # print(len(src), src[0].shape)
        for t in range(self.args.max_num_steps):
            src_tensor = torch.stack(src).squeeze(2)
            src_batch.append(src_tensor)
            mask_batch_all.append(torch.stack(masks))

        return state_batch, tgt_batch, src_batch, mask_batch, mask_batch_all

    def sample_k_traj_zeroshot(self, k_traj, policy_idx=0, env_idx=None):
        '''
        Sample a number of trajectories using the inferred dynamics 
        from a small number of interactions with a new environment.
        k_traj: number of episodes used to train the dynamics embedding
        '''
        trajectory = self.sample_policy(policy_idx=policy_idx)
        context_env = self.sample_env_context(trajectory, env_idx=env_idx)
        state_action_list = []
        target_list = []
        source_list = []

        for k in range(k_traj):
            eval_episode_rewards = []
            init_state = torch.tensor(self.env.reset(same=True)).float()
            state = init_state
            obs = init_state
            trajectory = self.sample_policy(policy_idx=policy_idx)
            for _ in range(self.inf_num_steps):     # 这个值为１，感觉也只是对环境不同体现在动态性上比较有效吧
                obs_feat = obs.float().squeeze().reshape(1,-1).to(torch.device(self.args.device))

                recurrent_hidden_state = torch.zeros(
                    1, trajectory.recurrent_hidden_state_size, device=self.args.device)
                mask = torch.zeros(1, 1, device=self.args.device)
                trajectory = trajectory.float()
                action = trajectory.act(
                            obs_feat, recurrent_hidden_state.float(), mask.float(),
                            deterministic=True)[1].detach()

                action_tensor = action.float().reshape(self.action_dim)
                state_action_tensor = torch.cat([obs.squeeze().to(device=self.args.device),
                    action_tensor.to(device=self.args.device)], dim=0)
                obs, reward, done, infos = self.env.step(*action.cpu().numpy())

                obs = torch.tensor(obs).float()
                target_tensor = obs

                state_action_list.append(state_action_tensor)
                target_list.append(target_tensor)
                source_list.append(context_env.reshape(self.args.max_num_steps, -1))

            return state_action_list, target_list, source_list


class EnvSamplerPDVF():
    '''
    Environment sampler object for training the 
    Policy-Dynamics Value Function.
    '''
    def __init__(self, env, base_policy, args):
        if args.env_name.startswith('spaceship'):
            self.env = env.venv.envs[0]
        elif args.env_name.startswith('myswimmer'):
            self.env = env.venv.envs[0].env.env
        else:
            self.env = env.venv.envs[0].env.env.env
        self.env.reset()
        self.env1 = env
        self.base_policy = base_policy
        self.args = args
        self.env_name = args.env_name
        self.action_dim = self.env1.action_space.shape[0]
        self.state_dim = self.env1.observation_space.shape[0]
        self.enc_input_size = 2 * self.state_dim + self.action_dim
        self.inf_num_steps = args.inf_num_steps
        self.env_max_seq_length = args.inf_num_steps * self.enc_input_size
        self.max_seq_length = args.max_num_steps * self.enc_input_size 

        if 'ant' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_ANT
        elif 'swimmer' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_SWIMMER
        elif 'spaceship' in args.env_name:
            self.max_episode_steps = MAX_EPISODE_STEPS_SPACESHIP

    def reset(self):
        '''
        Reset the environment to an environment that might have different dynamics.
        '''
        return self.env1.reset()

    def reset_same(self):
        '''
        Reset the environment to an environment with identical dynamics.
        '''
        return self.env.reset(same=True)

    def sample_policy(self, policy_idx=0):
        '''
        Sample a policy from your set of pretrained policies. 
        '''
        
        # print(len(self.base_policy))
        return self.base_policy[policy_idx]
        
    def generic_step(self, policy_fn, state):
        '''
        Take a step in the environment 
        '''
        recurrent_hidden_state = torch.zeros(
            1, policy_fn.recurrent_hidden_state_size, device=self.args.device)
        mask = torch.zeros(1, 1, device=self.args.device)
        policy_fn = policy_fn.float().to(self.args.device)
        action = policy_fn.act(
            state.squeeze().unsqueeze(0).to(torch.device(self.args.device)).float(),
            recurrent_hidden_state.float(), mask.float(),
            deterministic=True)[1].detach()
        action_tensor = action.float().reshape(self.action_dim)
        state_action_tensor = torch.cat([state.float().squeeze().to(self.args.device),
            action_tensor.to(self.args.device)], dim=0)
        next_state, reward, done, _ = self.env.step(*action.cpu().numpy())

        next_state_tensor = torch.Tensor(next_state).float().to(self.args.device)
        if not done:
            sas_tensor = torch.cat((state_action_tensor, next_state_tensor), dim=0).unsqueeze(1)
        else:
            sas_tensor = torch.FloatTensor([0 for _ in range(self.enc_input_size)]) \
                .reshape(self.enc_input_size, 1)

        res = {'next_state': next_state_tensor, 'action': action_tensor, 'sa_tensor': state_action_tensor,
               'sas_tensor': sas_tensor.to(self.args.device), 'reward': reward, 'done': done}
        return res

    def sample_env_context(self, policy_fn, env_idx=None):
        '''
        Generate a few steps in the new environment. 
        Use these transitions to infer the environment's dynamics.  
        '''
        src = []
        this_src_batch = []
        if env_idx is not None:
            state = self.env.reset(env_id=env_idx)
        else:
            state = self.reset_same()
        state = torch.tensor(state).float()
        done = False

        max_num_steps = self.inf_num_steps
        
        for t in range(max_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, sas_tensor = res['next_state'], res['done'], res['sas_tensor']
            src.append(sas_tensor)
            src_pad = F.pad(torch.stack(src).reshape((t + 1) * self.enc_input_size), \
                            (0, self.env_max_seq_length - (t + 1) * len(sas_tensor)))

            this_src_batch.append(src_pad)
            if done:
                break

        res = this_src_batch[-1].reshape(max_num_steps, -1)
        return res

    def zeroshot_sample_src_from_pol_state_ds(self, args, init_state, sizes, policy_idx=0, env_idx=0):
        '''
        Sample transitions using a certain policy and starting in a given state.
        Works for Spaceship.
        '''
        # get the policy embedding
        src_policy = []
        masks_policy = []
        src_env = []
        episode_reward = 0
        state = init_state
        k = 0
        policy_fn = self.sample_policy(policy_idx=policy_idx)
        for t in range(args.max_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                         res['sa_tensor'], res['sas_tensor']
            episode_reward = args.gamma * episode_reward + reward
            src_policy.append(sa_tensor)
            src_env.append(sas_tensor)
            k+=1
            masks_policy.append(torch.FloatTensor([done == False]))     # 把一个episode的最后一个state maks掉了
            if done:
                # print(k)
                break

        if self.env_name.startswith('myacrobot'):
            episode_reward += args.max_num_steps

        return src_policy, src_env, masks_policy, episode_reward

    def zeroshot_sample_src_from_pol_state(self, args, init_state, sizes, policy_idx=0, env_idx=0):
        '''
        Sample transitions using a certain policy and starting in a given state. 
        Works for Spaceship. 
        '''
        # get the policy embedding 
        src_policy = []
        masks_policy = []

        episode_reward = 0
        state = init_state
        policy_fn = self.sample_policy(policy_idx=policy_idx)
        for t in range(args.max_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                         res['sa_tensor'], res['sas_tensor']
            episode_reward = args.gamma * episode_reward + reward
            src_policy.append(sa_tensor)
            masks_policy.append(torch.FloatTensor([done == False]))     # 把一个episode的最后一个state maks掉了
            if done:
                break

        policy_feats = torch.stack(src_policy).unsqueeze(0)
        mask_policy = torch.stack(masks_policy).squeeze(1).unsqueeze(0).unsqueeze(0)

        if self.env_name.startswith('myacrobot'):
            episode_reward += args.max_num_steps

        # get the env embedding 
        src_env = []
        env_feats = []

        state = init_state
        policy_fn = self.sample_policy(policy_idx=policy_idx)
        for t in range(self.inf_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                         res['sa_tensor'], res['sas_tensor']
            env_feats.append(sas_tensor)
            env_pad = F.pad(torch.stack(env_feats).reshape((t+1) * self.enc_input_size), \
                             (0, self.env_max_seq_length - (t+1) * len(sas_tensor)))
            print(env_feats)
            print(env_pad.shape)
            if done:
                break
            
        source_env = torch.stack([env_pad.reshape(self.inf_num_steps,len(env_feats[0]))])
        mask_env = (source_env != 0).unsqueeze(-2)
        mask_env = mask_env[:, :, :, 0].squeeze(2).unsqueeze(1)

        res = {'source_env': source_env,
               'source_policy': policy_feats,
               'mask_policy': mask_policy,
               'mask_env': mask_env,
               'episode_reward': episode_reward,
               't': t,
               'init_state': init_state}

        return res
    
    def zeroshot_sample_src_from_pol_state_mujoco(self, args, init_state, sizes, policy_idx=0, env_idx=0):
        '''
        Sample transitions using a certain policy and starting in a given state. 
        Works for Swimmer and Ant.
        '''
        # get the policy embedding 
        src_policy = []
        masks_policy = []

        episode_reward = 0
        state = init_state
        policy_fn = self.sample_policy(policy_idx=policy_idx)
        for t in range(args.max_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                         res['sa_tensor'], res['sas_tensor']
            episode_reward = args.gamma * episode_reward + reward
            src_policy.append(sa_tensor)
            masks_policy.append(torch.FloatTensor([done == False]))
            if done:
                break
        if not done:
            for t in range(self.max_episode_steps - args.max_num_steps):
                res = self.generic_step(policy_fn, state)
                state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                            res['sa_tensor'], res['sas_tensor']
                episode_reward =args.gamma * episode_reward + reward
                if done:
                    break

        policy_feats = torch.stack(src_policy).unsqueeze(0)
        mask_policy = torch.stack(masks_policy).squeeze(1).unsqueeze(0).unsqueeze(0)

        # get the env embedding
        src_env = []
        env_feats = []

        state = init_state
        policy_fn = self.sample_policy(policy_idx=policy_idx)
        for t in range(self.inf_num_steps):
            res = self.generic_step(policy_fn, state)
            state, done, reward, sa_tensor, sas_tensor = res['next_state'], res['done'], res['reward'], \
                                                         res['sa_tensor'], res['sas_tensor']
            env_feats.append(sas_tensor)
            env_pad =  F.pad(torch.stack(env_feats).reshape((t+1) * self.enc_input_size), \
                             (0, self.env_max_seq_length - (t+1) * len(sas_tensor)))
            if done:
                break

        source_env = torch.stack([env_pad.reshape(self.inf_num_steps,  len(env_feats[0]))])
        mask_env = (source_env != 0).unsqueeze(-2)
        mask_env = mask_env[:, :, :, 0].squeeze(2).unsqueeze(1)
        
#        if args.norm_reward:
#            episode_reward = (episode_reward - args.min_reward) / (args.max_reward - args.min_reward)
        
        res = {'source_env': source_env,
               'source_policy': policy_feats,
               'mask_policy': mask_policy,
               'mask_env': mask_env,
               'episode_reward': episode_reward,
               't': t,
               'init_state': init_state}

        return res

    def get_reward_pol_embedding_state(self, args, init_state, init_obs, policy_emb, decoder, env_idx=0, verbose=False):
        '''
        Estimate the return using Monte-Carlo for a given policy embedding starting at a given initial state. 
        Works for Spaceship. 
        '''
        self.env.reset(env_id=env_idx)
        self.env.state = init_state
        state = init_obs

        device = args.device
        done = False
        episode_reward = 0
        for t in range(self.args.max_num_steps): # yuanlaishi args.
            recurrent_hidden_state = torch.zeros(policy_emb.shape[0],
                decoder.recurrent_hidden_state_size, device=device, requires_grad=True).float()
            mask_dec = torch.zeros(policy_emb.shape[0], 1, device=device, requires_grad=True).float()
            emb_state_input = torch.cat((policy_emb.to(device), state.unsqueeze(0).to(device)), dim=1).to(device)
            action = decoder.act(emb_state_input, recurrent_hidden_state, mask_dec,
                deterministic=True)[1].squeeze().cpu().detach().numpy()
            next_state, reward, done, _ = self.env.step(action)
            
            episode_reward = args.gamma * episode_reward + reward # yuanlaishi gamma
            state = torch.FloatTensor(next_state)
            if done:
                break
        return episode_reward, t
        
    def get_reward_pol_embedding_state_mujoco(self, args, init_state, init_obs, policy_emb, decoder, env_idx=0, verbose=False):
        '''
        Estimate the return using Monte-Carlo for a given policy embedding starting at a given initial state. 
        Works for Swimmer and Ant. 
        '''
        self.env.reset(env_id=env_idx)
        self.env.sim.set_state(init_state)
        state = init_obs 

        device = args.device
        done = False
        episode_reward = 0
        t = 0
        for t in range(self.max_episode_steps):
            t += 1
            recurrent_hidden_state = torch.zeros(policy_emb.shape[0],
                decoder.recurrent_hidden_state_size, device=device, requires_grad=True).float()
            mask_dec = torch.zeros(policy_emb.shape[0], 1, device=device, requires_grad=True).float()
            emb_state_input = torch.cat((policy_emb.to(device), state.unsqueeze(0).to(device)), dim=1).to(device)
            action = decoder.act(emb_state_input, recurrent_hidden_state, mask_dec,
                deterministic=True)[1].squeeze().cpu().detach().numpy()
            next_state, reward, done, _ = self.env.step(action)
            
            episode_reward = args.gamma * episode_reward + reward # yuanlaishi gamma
            state = torch.FloatTensor(next_state)
            if done: 
                break
        # print(episode_reward,'---------')    
        if args.norm_reward:
            episode_reward = (episode_reward - args.min_reward) / (args.max_reward - args.min_reward)
        # print(episode_reward,'---------11')
        return episode_reward, t

    def sample_policy_data(self, policy_idx=0, env_idx=None):
        '''
        Sample transitions from a given policy in your collection.
        '''
        state_batch = []
        tgt_batch = []
        src_batch = []
        mask_batch = []
        mask_batch_all = []

        if env_idx is not None:
            state = self.env.reset(env_id=env_idx)
        else:
            state = self.env.reset()

        trajectory = self.sample_policy(policy_idx=policy_idx)
        state = torch.tensor(state).unsqueeze(0)  
        res = self.generic_step(trajectory, state)
        state, action, done, reward, sa_tensor = res['next_state'], res['action'], res['done'], res['reward'], \
                                                     res['sa_tensor']
        sa_tensor = torch.transpose(sa_tensor, 1, 0)
        src = []
        masks = []

        for t in range(self.args.max_num_steps):
            state_batch.append(state)
            src.append(sa_tensor)
            tgt_batch.append(action.detach())
            res = self.generic_step(trajectory, state)
            state, action, done, reward, sa_tensor = res['state'], res['action'], res['done'], res['reward'], \
                                                                    res['sa_tensor']

            sa_tensor = torch.transpose(sa_tensor, 1, 0)
            masks.append(torch.FloatTensor([done == False]))
            mask_batch.append(torch.FloatTensor([done == False])) 

        for t in range(self.args.max_num_steps):
            src_tensor = torch.stack(src).squeeze(2)
            src_batch.append(src_tensor)
            mask_batch_all.append(torch.stack(masks))

        return state_batch, tgt_batch, src_batch, mask_batch, mask_batch_all
