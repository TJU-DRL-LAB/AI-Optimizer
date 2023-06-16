# Policy Expansion (PEX)  [ICLR 2023]

## Installation
The training environment (PyTorch and dependencies) can be installed as follows:

```
git clone https://github.com/Haichao-Zhang/PEX.git
cd PEX

python3 -m venv .venv_pex
source .venv_pex/bin/activate
pip3 install -e .
```

## Train

### Offline Training

Set ```root_dir``` to the path where the experimental results will be saved.

Then run:

```
CUDA_VISIABLE_DEVICES=0 python main_offline.py --log_dir=$root_dir/antmaze-large-play-v0_offline_run1 --env_name antmaze-large-play-v0 --tau 0.9 --beta 10.0
```

### Online Training
First set the path to the offline checkpoint:
```
path_to_offline_ckpt=$root_dir/antmaze-large-play-v0_offline_run1/offline_ckpt
```

and select an algorithm:
```
algorithm=pex (or any other algorithms in [scratch, direct, buffer, pex])
```

and then run
```
CUDA_VISIABLE_DEVICES=0 python ./main_online.py --log_dir=$root_dir/antmaze-large-play-v0_run1_$algorithm --env_name=antmaze-large-play-v0 --tau 0.9 --beta 10.0 --ckpt_path=$path_to_offline_ckpt --eval_episode_num=100 --algorithm=$algorithm
```


### Example on Locomotion Task

```
CUDA_VISIABLE_DEVICES=0 python main_offline.py --log_dir=$root_dir/halfcheetah-random-v2_offline_run1 --env_name halfcheetah-random-v2 --tau 0.9 --beta 10.0

path_to_offline_ckpt=$root_dir/halfcheetah-random-v2/offline_ckpt

CUDA_VISIABLE_DEVICES=0 python ./main_online.py --log_dir=$root_dir/halfcheetah-random-v2_run1_$algorithm --env_name=halfcheetah-random-v2 --tau 0.9 --beta 10.0 --ckpt_path=$path_to_offline_ckpt --eval_episode_num=10 --algorithm=$algorithm
```


## Paper

<b>[Policy Expansion for Bridging Offline-to-Online Reinforcement Learning](https://arxiv.org/pdf/2302.00935.pdf)</b> <br>

[Haichao Zhang](https://sites.google.com/site/hczhang1/),
Wei Xu,
Haonan Yu

*International Conference on Learning Representations* (ICLR), 2023



## Cite

Please cite our work if you find it useful:

```
@inproceedings{PEX,
  author    = {Haichao Zhang and Wei Xu and Haonan Yu},
  title     = {Policy Expansion for Bridging Offline-to-Online Reinforcement Learning},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2023},
}
```
