# Ensemble-based Offline-to-Online Reinforcement Learning: From Pessimistic Learning to Optimistic Exploration (E2O)

## Installation
The training environment (PyTorch and dependencies) can be installed as follows:

```
git clone https://github.com/.git
```

## Train

### Offline Training

```
nohup python E2O-offline.py --dataset='halfcheetah-medium-expert-v2' --n_critic=10 --gpu=0 --seed=1 >output_E2O-CQL-10_halfcheetah-medium-expert-v2_seed$seed.txt 2>&1 &
```

### Online Training

```
nohup python E2O-online.py --env='HalfCheetah-v2' --gpu=0 --seed=1 >output_E2O-CQL-10_online_HalfCheetah-v2-medium-expert_seed$seed.txt 2>&1 &
```


## Paper

<b>[Ensemble-based Offline-to-Online Reinforcement Learning: From Pessimistic Learning to Optimistic Exploration](https://arxiv.org/pdf/2306.06871.pdf) <br>


## Cite

Please cite our work if you find it useful:

```
@article{zhao2023ensemble,
  title={Ensemble-based Offline-to-Online Reinforcement Learning: From Pessimistic Learning to Optimistic Exploration},
  author={Zhao, Kai and Ma, Yi and Liu, Jinyi and Zheng, Yan and Meng, Zhaopeng},
  journal={arXiv preprint arXiv:2306.06871},
  year={2023}
}
```
