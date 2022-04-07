import argparse

import torch
import math

def get_args():
    parser = argparse.ArgumentParser(description='PPO')

    # PPO
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='number of CPU processes to use for training')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2048, 
        help='number of forward steps in PPO')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo ')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='eval interval, one eval per n updates')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=3e6,
        help='number of environment steps to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size used for training the embeddings')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate ')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one checkpoint per n updates')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon')
    parser.add_argument('--log-dir', default='./logs',
                        help='directory to save agent logs (..usfa-marl-data/logs)')
    parser.add_argument('--save-dir', default='./models/ppo-policies/',
                        help='directory to save models')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--env-name', type=str, default='spaceship-v0', help='environment')
    parser.add_argument('--default-ind', type=int, default=0,
                        help='default ind for train envs')
    parser.add_argument('--num-envs', type=int, default=20,
                        help='total number of environments both trian and test')
    parser.add_argument('--basepath', type=str, 
                        default='/home/roberta/miniconda3/envs/pdvf_icml/lib/python3.7/site-packages/gym/envs/mujoco/assets/',
                        help='path to xml files')    
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value to clip gradient norm')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else 

    return args
