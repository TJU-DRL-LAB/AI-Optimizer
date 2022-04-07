import os
import torch
import embedding_networks
from ppo.model import Policy


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def save_model(prefix, model, optimizer, num_epoch, args, suffix=None,
               policy_embedding=True, save_dir=None):
    '''
        Save a pretrained model for later use.
    '''
    if not save_dir:
        return
    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    save_dict = {
        'num_epoch': num_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dict['args'] = vars(args)

    name = prefix
    if suffix:
        name += suffix
    if not name.endswith('.pt'):
        name += '.pt'

    save_path = os.path.join(save_dir, "%s" % (name))
    torch.save(save_dict, save_path)

    return save_path


def torch_load(path, device):
    '''
    Load a generic pretrained model.
    '''
    return torch.load(path, map_location=device)


def load_policy_model(args, env):
    '''
    Load a pretrined policy embedding model.
    '''
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_encoder_dim = args.num_attn_heads * args.policy_attn_head_dim
    policy_enc_input_size = state_dim + action_dim

    policy_encoder = embedding_networks.make_encoder_oh(policy_enc_input_size, N=args.num_layers, \
                                                        d_model=policy_encoder_dim, h=args.num_attn_heads,
                                                        dropout=args.dropout, \
                                                        d_emb=args.policy_embedding_dim, use_extra_fc=True,
                                                        no_norm=False)
    print("policy model: "+ str(args.seed)+'-'+ str(args.num_epochs_emb_policy)+'-'+str(args.policy_embedding_dim)+'-'+str(args.num_t_policy_embed)+'-'+str(args.shuffle))
    policy_encoder_model = str(args.seed)+'-'+ str(args.num_epochs_emb_policy)+'-'+str(args.policy_embedding_dim)+'-'+str(args.num_t_policy_embed)+'-'+str(args.shuffle)+'-norm-cont-policy-encoder.{}.pt'.format(args.env_name)
    # '0-1000-8-20-1-norm-cont-policy-encoder.{}.pt'.format(args.env_name)
    # policy_decoder_model = 'policy-decoder.{}.pt'.format(args.env_name)

    policy_encoder_path = os.path.join(args.save_dir_policy_embedding, policy_encoder_model)
    policy_encoder_checkpoint = torch_load(policy_encoder_path, args.device)
    policy_encoder.load_state_dict(policy_encoder_checkpoint['state_dict'])
    policy_encoder.to(args.device)
    policy_encoder.eval()

    # policy_decoder = Policy(
    #         tuple([env.observation_space.shape[0] + args.policy_embedding_dim]),
    #         env.action_space,
    #         base_kwargs={'recurrent': False})
    # policy_decoder_path = os.path.join(args.save_dir_policy_embedding, policy_decoder_model)
    # policy_decoder_checkpoint = torch_load(policy_decoder_path, args.device)
    # policy_decoder.load_state_dict(policy_decoder_checkpoint['state_dict'])
    # policy_decoder.to(args.device)
    # policy_decoder.eval()

    return policy_encoder  # , policy_decoder


def load_dynamics_model(args, env):
    '''
    Load a pretrined dynamics / environment embedding model.
    '''
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env_encoder_dim = args.num_attn_heads * args.dynamics_attn_head_dim
    env_enc_input_size = 2 * state_dim + action_dim
    env_encoder_model = str(args.seed)+'-'+ str(args.num_epochs_emb_env)+'-'+str(args.dynamics_embedding_dim)+'-'+str(args.num_t_env_embed)+'-'+str(args.shuffle)+'-nonorm-dynamics-encoder.{}.pt'.format(args.env_name)
    # env_encoder_model = '0-800-64-1-1-nonorm-dynamics-encoder.{}.pt'.format(args.env_name)
    # env_decoder_model = 'dynamics-decoder.{}.pt'.format(args.env_name)
    print("env model: "+str(args.seed)+'-'+ str(args.num_epochs_emb_env)+'-'+str(args.dynamics_embedding_dim)+'-'+str(args.num_t_env_embed)+'-'+str(args.shuffle))
    env_encoder = embedding_networks.make_encoder_oh(env_enc_input_size, N=args.num_layers, \
                                                     d_model=env_encoder_dim, h=args.num_attn_heads,
                                                     dropout=args.dropout, \
                                                     d_emb=args.dynamics_embedding_dim, use_extra_fc=True,
                                                     no_norm=False)
    env_encoder_path = os.path.join(args.save_dir_dynamics_embedding, env_encoder_model)
    env_encoder_checkpoint = torch_load(env_encoder_path, args.device)
    env_encoder.load_state_dict(env_encoder_checkpoint['state_dict'])
    env_encoder.to(args.device)
    env_encoder.eval()

    return env_encoder
