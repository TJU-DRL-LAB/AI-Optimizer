# ICLR 2022 GPL Workshop-Policy Adaptation via Decoupled Policy and Environment Representations (PAnDR)

This is the official implementation of 
our work [PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations](https://arxiv.org/abs/2204.02877)
presented at ICLR 2022 Workshop on Generalizable Policy Leanring (GPL).

## Introduction

Deep Reinforcement Learning (DRL) has been a promising solution to many complex decision-making problems. Nevertheless, the notorious weakness in generalization among environments prevent widespread application of DRL agents in real-world scenarios. 
Although advances have been made recently, most prior works assume sufficient online interaction on training environments, which can be costly in practical cases. 
To this end, we focus on an **offline-training-online-adaptation** setting, in which the agent first learns from offline experiences collected in environments with different dynamics and then performs online policy adaptation in environments with new dynamics. 

In this paper, we propose Policy Adaptation with Decoupled Representations (PAnDR) for fast policy adaptation. 
- In offline training phase, the **environment representation** and **policy representation** are learned through _contrastive learning_ and _policy recovery_, respectively. The representations are further refined by _mutual information optimization_ to make them **more decoupled and complete**. With learned representations, a Policy-Dynamics Value Function (PDVF) (Raileanu et al., 2020) network is trained to approximate the values for different combinations of policies and environments. 
- In online adaptation phase, with the environment context inferred from few experiences collected in new environments, the policy is optimized by gradient ascent with respect to the PDVF.

A conceptual illustration is shown below.

<div align=center><img align="center" src="./assets/PAnDR_concept.png" alt="PAnDR Conceptual Illustration" style="zoom:40%;" /></div>

## Repo Content

### Folder Description


### Domains and Environments


## Installation

We recommend the user to install **anaconada** and or **venv** for convenient management of different python envs.

### Dependencies

- Python 



### Environment Installation
```
cd myant 
pip install -e .  

cd myswimmer 
pip install -e .  

cd myspaceship 
pip install -e .  
```

## Example Usage

### (1) Reinforcement Learning Phase 

Train PPO policies on each environments.

Each of the commands below need to be run 
for seed in [0,...,4] and for default-ind in [0,...,19].

#### Spaceship
```
python ppo/ppo_main.py \
--env-name spaceship-v0 --default-ind 0 --seed 0 
```

#### Swimmer
```
python ppo/ppo_main.py \
--env-name myswimmer-v0 --default-ind 0 --seed 0 
```

#### Ant-wind
```
python ppo/ppo_main.py \
--env-name myant-v0 --default-ind 0 --seed 0 
```

### (2) PAnDR Training Phase

#### Ant-wind
```
# python main_train.py --env-name myant-v0 --op-lr 0.01 --num-t-policy-embed 50 --num-t-env-embed 1 --gd-iter 50 --norm-reward --min-reward -200 --max-reward 1000 --club-lambda 1000 --mi-lambda 1
```


We refer the user to our paper for complete details of hyperparameter settings and design choices.

## TO-DO
- [ ] Tidy up redundant codes

## Citation
If this repository has helped your research, please cite the following:
```bibtex
@inproceedings{sang2022pandr,
  title     = {PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations},
  author    = {Tong Sang, Hongyao Tang, Yi Ma, Jianye Hao, Yan Zheng, Zhaopeng Meng, Boyan Li, Zhen Wang},
  booktitle = {International Conference on Learning Representations Workshop on Generalizable Policy Learning},
  year      = {2022},
  url       = {https://arxiv.org/abs/2204.02877}
}
```

