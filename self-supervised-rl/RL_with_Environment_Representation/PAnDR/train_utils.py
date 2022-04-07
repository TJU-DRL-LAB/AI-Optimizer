import torch
import torch.nn as nn
import torch.optim as optim
from pandr_storage import TransitionPDVF_vl, TransitionPolicyDecoder, TransitionPDVF
import random
import numpy as np
import torch.nn.functional as F


def optimize_model_cl(args, network, memory, env_encoder, policy_encoder, num_opt_steps=1, eval=False, vl_train=False):
   '''
   Train the Policy-Dynamics Value Function on the initial dataset (phase 1).
   '''
   if len(memory) < args.batch_size_pdvf:
       return

   device = args.device
   value_net = network['net']
   optimizer = network['optimizer']
   l2_loss = nn.MSELoss()

   total_loss = 0
   # print(len(memory))
   for _ in range(num_opt_steps):
       transitions = memory.sample(args.batch_size_pdvf)
       batch = TransitionPDVF_vl(*zip(*transitions))
       state_batch = torch.cat(batch.state).squeeze(1)
       emb_policy_batch = []
       emb_env_batch = []
       for i in range(args.batch_size_pdvf):
           policy_data = batch.policy_data[i]

           policy_mask = batch.policy_mask[i]
           env_data = batch.env_data[i]
           env_mask = batch.env_mask[i]
           p = torch.FloatTensor(policy_data).to(device)
           e = torch.FloatTensor(env_data).to(device)
           pm = torch.FloatTensor(policy_mask).to(device)
           em = torch.FloatTensor(env_mask).to(device)
           policy_data = p.squeeze(-1).unsqueeze(0)

           env_data = e.squeeze(-1).unsqueeze(0)
           policy_mask = pm.squeeze(-1).unsqueeze(0).unsqueeze(0)

           env_mask = em.unsqueeze(0)
           if env_data.shape[1] == 1:
               env_data = env_data.repeat(1, 2, 1)
               env_mask = env_mask.repeat(1, 1, 2)
           emb_policy = policy_encoder.infer_posterior(policy_data.detach().to(device),
                                       policy_mask.detach().to(device)).detach()
           emb_env = env_encoder.infer_posterior(env_data.detach().to(device),
                                 env_mask.detach().to(device)).detach()

           emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
           emb_env = F.normalize(emb_env, p=2, dim=1).detach()
           # print(emb_policy, emb_env)
           emb_policy_batch.extend(emb_policy)
           emb_env_batch.extend(emb_env)

           # print(len(batch.total_return),batch.total_return[0].shape)
       emb_policy_batch = torch.stack(emb_policy_batch)
       emb_env_batch = torch.stack(emb_env_batch)
       total_return_batch = torch.cat(batch.total_return)
       # print(state_batch.shape, emb_policy_batch.shape,emb_env_batch.shape, total_return_batch.shape)
       state_values = value_net(state_batch.to(device).detach(),
       emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

       loss = l2_loss(state_values, total_return_batch.unsqueeze(1))
       total_loss += loss.item()
       if not eval:
           if vl_train:
               env_encoder.encoder_optimizer.zero_grad()
               policy_encoder.encoder_optimizer.zero_grad()
               optimizer.zero_grad()
               loss.backward()
               env_encoder.encoder_optimizer.step()
               policy_encoder.encoder_optimizer.step()
               optimizer.step()
           else:
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
           # print("traing----------------------")


   return total_loss

def optimize_model_pdvf(args, network, memory, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the initial dataset (phase 1).
    '''
    if len(memory) < args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        transitions = memory.sample(args.batch_size_pdvf)
        batch = TransitionPDVF(*zip(*transitions))
        state_batch = torch.cat(batch.state).squeeze(1)
        emb_policy_batch = torch.cat(batch.emb_policy).squeeze(1)
        emb_env_batch = torch.cat(batch.emb_env).squeeze(1)
        total_return_batch = torch.cat(batch.total_return)

        state_values = value_net(state_batch.to(device).detach(),
                                 emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()
        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss

def optimize_model_pdvf_phase2(args, network, memory1, memory2, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the aggregated dataset 
    that includes both the best policy embeddings found in phase 1 
    and the original dataset (phase 2). 
    '''
    if len(memory1) < 1/4 * args.batch_size_pdvf:
        return
    if len(memory2) < 3/4 * args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer2']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        
        transitions1 = memory1.sample(int(1/4 * args.batch_size_pdvf))
        batch1= TransitionPDVF(*zip(*transitions1))
        state_batch1 = torch.cat(batch1.state).squeeze(1)
        emb_policy_batch1 = torch.cat(batch1.emb_policy).squeeze(1)
        emb_env_batch1 = torch.cat(batch1.emb_env).squeeze(1)
        total_return_batch1 = torch.cat(batch1.total_return)

        transitions2 = memory2.sample(int(3/4 * args.batch_size_pdvf))
        batch2= TransitionPDVF(*zip(*transitions2))
        state_batch2 = torch.cat(batch2.state).squeeze(1)
        emb_policy_batch2 = torch.cat(batch2.emb_policy).squeeze(1)
        emb_env_batch2 = torch.cat(batch2.emb_env).squeeze(1)
        total_return_batch2 = torch.cat(batch2.total_return)
        
        state_batch = torch.cat([state_batch1, state_batch2], dim=0)
        emb_policy_batch = torch.cat([emb_policy_batch1, emb_policy_batch2], dim=0)
        emb_env_batch = torch.cat([emb_env_batch1, emb_env_batch2], dim=0)
        total_return_batch = torch.cat([total_return_batch1, total_return_batch2], dim=0)

        indices = [i for i in range(state_batch.shape[0])]
        random.shuffle(indices)

        state_batch = state_batch[indices]
        emb_policy_batch = emb_policy_batch[indices]
        emb_env_batch = emb_env_batch[indices]
        total_return_batch = total_return_batch[indices]
        
        state_values = value_net(state_batch.to(device).detach(), 
            emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
 
    return total_loss

def optimize_model_pdvf(args, network, memory, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the initial dataset (phase 1).
    '''
    if len(memory) < args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        transitions = memory.sample(args.batch_size_pdvf)
        batch = TransitionPDVF(*zip(*transitions))
        state_batch = torch.cat(batch.state).squeeze(1)
        emb_policy_batch = torch.cat(batch.emb_policy).squeeze(1)
        emb_env_batch = torch.cat(batch.emb_env).squeeze(1)
        # print(len(batch.total_return),batch.total_return[0].shape)
        total_return_batch = torch.cat(batch.total_return)
        # print(emb_policy_batch.shape, emb_env_batch.shape, state_batch.shape, total_return_batch.shape)
        state_values = value_net(state_batch.to(device).detach(),
                                 emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())
        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()
        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("traing----------------------")

    return total_loss


def optimize_model_pdvf_phase2_cl(args, network, memory1, memory2, env_encoder, policy_encoder, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the aggregated dataset
    that includes both the best policy embeddings found in phase 1
    and the original dataset (phase 2).
    '''
    if len(memory1) < 1 / 4 * args.batch_size_pdvf:
        return
    if len(memory2) < 3 / 4 * args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer2']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):

        transitions1 = memory1.sample(int(1 / 4 * args.batch_size_pdvf))
        batch1 = TransitionPDVF_vl(*zip(*transitions1))
        state_batch1 = torch.cat(batch1.state).squeeze(1)
        emb_policy_batch1 = []
        emb_env_batch1 = []

        policy_data = batch1.policy_data

        policy_mask = batch1.policy_mask
        env_data = batch1.env_data
        env_mask = batch1.env_mask
        p = torch.FloatTensor(policy_data).to(device)
        e = torch.FloatTensor(env_data).to(device)
        pm = torch.FloatTensor(policy_mask).to(device)
        em = torch.FloatTensor(env_mask).to(device)
        policy_data = p.squeeze(-1)

        env_data = e.squeeze(-1)
        policy_mask = pm.squeeze(-1).unsqueeze(1)

        env_mask = em
        if env_data.shape[1] == 1:
            env_data = env_data.repeat(1, 2, 1)
            env_mask = env_mask.repeat(1, 1, 2)

        emb_policy = policy_encoder.infer_posterior(policy_data.detach().to(device),
                                                    policy_mask.detach().to(device)).detach()
                                                    
        emb_env = env_encoder.infer_posterior(env_data.detach().to(device),
                                              env_mask.detach().to(device)).detach()

        emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
        emb_env = F.normalize(emb_env, p=2, dim=1).detach()
        emb_policy_batch1.extend(emb_policy)
        emb_env_batch1.extend(emb_env)


        emb_policy_batch1 = torch.stack(emb_policy_batch1)
        emb_env_batch1 = torch.stack(emb_env_batch1)
        total_return_batch1 = torch.cat(batch1.total_return)

        transitions2 = memory2.sample(int(3 / 4 * args.batch_size_pdvf))
        batch2 = TransitionPDVF(*zip(*transitions2))
        state_batch2 = torch.cat(batch2.state).squeeze(1)
        emb_policy_batch2 = torch.cat(batch2.emb_policy).squeeze(1)
        emb_env_batch2 = torch.cat(batch2.emb_env).squeeze(1)
        total_return_batch2 = torch.cat(batch2.total_return)

        state_batch = torch.cat([state_batch1, state_batch2], dim=0)
        emb_policy_batch = torch.cat([emb_policy_batch1, emb_policy_batch2], dim=0)
        emb_env_batch = torch.cat([emb_env_batch1, emb_env_batch2], dim=0)
        total_return_batch = torch.cat([total_return_batch1, total_return_batch2], dim=0)

        indices = [i for i in range(state_batch.shape[0])]
        random.shuffle(indices)

        state_batch = state_batch[indices]
        emb_policy_batch = emb_policy_batch[indices]
        emb_env_batch = emb_env_batch[indices]
        total_return_batch = total_return_batch[indices]

        state_values = value_net(state_batch.to(device).detach(),
                                 emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    return total_loss


def optimize_decoder(args, network, memory, num_opt_steps=1, eval=False):
    '''
    Train the Policy Decoder on the original dataset (phase 1).
    '''
    if len(memory) < args.batch_size_pdvf:
        return

    device = args.device
    decoder = network['policy_decoder']
    optimizer = network['decoder_optimizer']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):

        transitions = memory.sample(args.batch_size_pdvf)
        batch = TransitionPolicyDecoder(*zip(*transitions))
        emb_state_batch = torch.cat(batch.emb_state).squeeze(1)
        recurrent_state_batch = torch.cat(batch.recurrent_state)
        mask_batch = torch.cat(batch.mask)
        action_batch = torch.cat(batch.action)

        pred_action = decoder(emb_state_batch.to(device).detach(), \
                              recurrent_state_batch.to(device).detach(), \
                              mask_batch.to(device).detach())

        loss = l2_loss(pred_action, action_batch.detach())
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss


def optimize_decoder_phase2(args, network, memory1, memory2, num_opt_steps=1, eval=False):
    '''
    Train the Policy Decoder on the an aggregated dataset containing the
    states and policy embeddings generated in phase 1 of training the PD-VF
    and the original dataset (phase 2).
    '''
    if len(memory1) < 1 / 4 * args.batch_size_pdvf:
        return
    if len(memory2) < 3 / 4 * args.batch_size_pdvf:
        return

    device = args.device
    decoder = network['policy_decoder']
    optimizer = network['decoder_optimizer2']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        transitions1 = memory1.sample(int(1 / 4 * args.batch_size_pdvf))
        batch1 = TransitionPolicyDecoder(*zip(*transitions1))
        emb_state_batch1 = torch.cat(batch1.emb_state).squeeze(1)
        recurrent_state_batch1 = torch.cat(batch1.recurrent_state)
        mask_batch1 = torch.cat(batch1.mask)
        action_batch1 = torch.cat(batch1.action)

        transitions2 = memory2.sample(int(3 / 4 * args.batch_size_pdvf))
        batch2 = TransitionPolicyDecoder(*zip(*transitions2))
        emb_state_batch2 = torch.cat(batch2.emb_state).squeeze(1)
        recurrent_state_batch2 = torch.cat(batch2.recurrent_state)
        mask_batch2 = torch.cat(batch2.mask)
        action_batch2 = torch.cat(batch2.action)

        emb_state_batch = torch.cat([emb_state_batch1, emb_state_batch2], dim=0)
        recurrent_state_batch = torch.cat([recurrent_state_batch1, recurrent_state_batch2], dim=0)
        mask_batch = torch.cat([mask_batch1, mask_batch2], dim=0)
        action_batch = torch.cat([action_batch1, action_batch2], dim=0)

        indices = [i for i in range(emb_state_batch.shape[0])]
        random.shuffle(indices)

        emb_state_batch = emb_state_batch[indices]
        recurrent_state_batch = recurrent_state_batch[indices]
        mask_batch = mask_batch[indices]
        action_batch = action_batch[indices]

        pred_action = decoder(emb_state_batch.to(device).detach(), \
                              recurrent_state_batch.to(device).detach(), \
                              mask_batch.to(device).detach())

        loss = l2_loss(pred_action, action_batch.detach())
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    return total_loss