import argparse
import torch
import math
import sys

def get_args():
    parser = argparse.ArgumentParser(description='PDVF')

    # PPO
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
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size used for training the embeddings')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--log-interval', type=int, default=50,
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
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--env-name', type=str, default='myant-v0', help='environment')
    parser.add_argument('--basepath', type=str, 
                        default='/home/st/anaconda3/envs/cuda11/lib/python3.7/site-packages/gym/envs/mujoco/assets/',
                        help='path to mujoco xml files')    
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value to clip gradient norm')
    
    # PDVF
    parser.add_argument('--use-mi-min', type=int, default=1)
    parser.add_argument('--use-mi-max', type=int, default=1)
    parser.add_argument('--op-lr',type=float, default=1)
    parser.add_argument('--decoder-training-step',type=int, default=3000)
    parser.add_argument('--use-pre-train', type=int, default=0)
    parser.add_argument('--mi-lambda', type=int, default=1)
    parser.add_argument('--club-lambda', type=int, default=1)
    parser.add_argument('--max-mutual-information', type=int, default=1)
    parser.add_argument('--data-set', type=int, default=1)
    parser.add_argument('--kl-lambda', type=float, default=1.0)
    parser.add_argument('--MI-lambda', type=float, default=5.0)
    parser.add_argument('--value-loss-train-embed', type=int, default=0)
    parser.add_argument('--use-information-bottleneck', type=int, default=0)
    parser.add_argument('--embedding-init-training-step', type=int, default=1000)
    parser.add_argument('--batch-size-every-test', type=int, default=30)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--batch-size-every-train', type=int, default=30)
    parser.add_argument('--zero-pad', type=int, default=0,
                        help='if use zero to pad dynamic transitions')
    parser.add_argument('--random-sample', type=int, default=0,
                        help='if random sample transition to form the input embedding')  # embedding需要很多个transition嘛，是完全随机采样，还是直接每多少个截断一下

    parser.add_argument('--batch-size-every-policy', type=int, default=1,
                        help='batchsize for each policy when training embedding network')
    parser.add_argument('--batch-size-every-env', type=int, default=1,
                        help='batchsize for each env when training embedding network')
    parser.add_argument('--gd-iter', type=int, default=10,
                        help='gradient descent iterations')
    parser.add_argument('--batch-size-every-z', type=int, default=1,
                        help='batchsize for each env and policy when training embedding network')
    parser.add_argument('--num-epochs-emb-env', type=int, default=1500,
                        help='num training step for env embedding')
    parser.add_argument('--num-epochs-emb-policy', type=int, default=1500,
                        help='num training step for policy embedding')
    parser.add_argument('--num-epochs-emb-z', type=int, default=1500,
                        help='num training step for z embedding')
    parser.add_argument('--num-epochs-emb-ep', type=int, default=1500,
                        help='num training step for policy and env embedding')
    parser.add_argument('--num-t-env-embed', type=int, default=1,
                        help='num transition step for embedding env')
    parser.add_argument('--num-t-both-embed', type=int, default=50,
                        help='num transition step for embedding env and policy')
    parser.add_argument('--num-t-policy-embed', type=int, default=50,
                        help='num transition step for embedding policy')
    parser.add_argument('--stage', type=int, default=20,
                        help='stage of training for evaluating the PDVF')
    parser.add_argument('--margin', type=int, default=0.0, # 1229
                        help='stage of training for evaluating the PDVF')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='normalize the reward when training the PDVF to be between 0 and 1 \
                        -- needed for the quadratic optimization')
    parser.add_argument('--min-reward', type=float, default=-400,
                        help='minimum reward used for normalization')
    parser.add_argument('--max-reward', type=float, default=1000,
                        help='maximum reward used for normalization')
    parser.add_argument('--dynamics-batch-size', type=int, default=32,
                        help='batch size used for training the dynamics embedding')
    parser.add_argument('--both-batch-size', type=int, default=64,
                        help='batch size used for training the dynamics embedding')
    parser.add_argument('--policy-batch-size', type=int, default=128, # change 2048
                        help='batch size used for training the policy embedding')
    parser.add_argument('--policy-embedding-states-size', type=int, default=20, # 1229
                        help='the number of states to embedding z_pai')
    parser.add_argument('--inf-num-steps', type=int, default=1, 
                        help='number of interactions with the new environment \
                        used to infer the dynamics embeddding')
    parser.add_argument('--num-train-eps', type=int, default=1,
                        help='number of episodes for collect training set')
    parser.add_argument('--num-eval-eps', type=int, default=3,
                        help='number of episodes for collect eval set with training env&pi')
    parser.add_argument('--num-stages', type=int, default=20, 
                        help='number of stages for training the PDVF')
    parser.add_argument('--num-epochs-pdvf-phase1', type=int, default=200,
                        help='number of epochs for training the PDVF in phase 1')
    parser.add_argument('--num-epochs-pdvf-phase2', type=int, default=100,
                        help='number of epochs for training the PDVF in phase 2')
    parser.add_argument('--num-envs', type=int, default=20, # change 20
                        help='total number of environments (both train and test)')
    parser.add_argument('--default-ind', type=int, default=0,
                        help='default index for the train envs')
    parser.add_argument('--lr-dynamics', type=float, default=0.001,
                        help='learning rate for the dynamics embedding')  
    parser.add_argument('--lr-policy', type=float, default=0.01,
                        help='learning rates for the policy embedding')
    parser.add_argument('--num-eps-dynamics', type=int, default=100,
                        help='number of episodes used to train the dynamics embedding')  
    parser.add_argument('--num-eps-policy', type=int, default=100, # change 20
                        help='number of episodes used to train the policy embedding')  # 采样时每个policy采样多少次
    parser.add_argument('--lr-pdvf', type=float, default=0.005,
                        help='learning rate for training the PDVF')
    parser.add_argument('--lr-q-distribution', type=float, default=0.005,
                        help='learning rate for training the q distribution')
    parser.add_argument('--save-dir-policy-embedding', \
                        default='./models/policy-embeddings-shared/',
                        help='directory to save models the policy embedding models')
    parser.add_argument('--save-dir-dynamics-embedding', \
                        default='./models/dynamics-embeddings-shared/',
                        help='directory to save models the dynamics embedding models')
    parser.add_argument('--save-dir-pdvf', \
                        default='../gvf_logs/pdvf-policies/',
                        help='directory to save models the PDVF')
    parser.add_argument('--batch-size-pdvf', type=int, default=128,
                        help='batch size used for training the PDVF')
    parser.add_argument('--hidden-dim-pdvf', type=int, default=128,
                        help='dimension of the hidden layers for the PDVF network')
    parser.add_argument('--num-opt-steps', type=int, default=1,
                        help='number of optimization steps to \
                        find the optimal policy of the PDVF')
    parser.add_argument('--num-seeds', type=int, default=5, 
                        help='number of seeds used to collect \
                        PPO policies in all the environments')
    
    # Embeddings
    parser.add_argument('--num-epochs-emb', type=int, default=200, # change 200
                        help='number of epochs for training the embeddings')
    parser.add_argument('--num-dec-traj', type=int, default=10,
                        help='number of trajectories to train the embeddings')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of layers in the encoder (with self attention)')
    parser.add_argument('--num-attn-heads', type=int, default=1,
                        help='number of attention heads')    
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout for use with rnn + attention decoder in seq2seq \
                        embedding model')    

    # Spaceship 
    parser.add_argument('--max-num-steps', type=int, default=50,
                        help='maximum number of steps allowed in the environment')
    parser.add_argument('--policy-embedding-dim', type=int, default=8,
                        help='dimension of the policy embedding')
    parser.add_argument('--dynamics-embedding-dim', type=int, default=8,
                        help='dimension of the dynamics embedding')
    parser.add_argument('--both-embedding-dim', type=int, default=8,
                        help='dimension of the env and policy embedding')
    parser.add_argument('--policy-hidden-dim-cond-policy', type=int, default=32,
                        help='hidden dimension of the policy autoencoder')
    parser.add_argument('--dynamics-hidden-dim-cond-policy', type=int, default=32,
                        help='hidden dimension of the environment autoencoder')
    parser.add_argument('--policy-attn-head-dim', type=int, default=64,
                        help='dimension of the policy attention head')
    parser.add_argument('--dynamics-attn-head-dim', type=int, default=64,
                        help='dimension of the dynamics attention head')
    parser.add_argument('--both-attn-head-dim', type=int, default=64,
                        help='dimension of the dynamics attention head')
    parser.add_argument('--z-hidden-dim', type=int, default=64,
                        help='dimension of approximate z distribution hidden network')

    # parse all arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
