# In-Sample Policy Iteration for Offline Reinforcement Learning

ISPI is a novel algorithm that by continuously refining the policy used for behavior regularization, 
in-sample policy iteration gradually improves itself while implicitly avoiding querying 
out-of-sample actions to avert catastrophic learning failures. Our theoretical analysis verifies its 
ability to learn the in-sample optimal policy, exclusively utilizing actions well-covered by the dataset.

## Installation
The training environment (PyTorch and dependencies) can be installed as follows:

```
git clone git@github.com:TJU-DRL-LAB/offline-rl-algorithms/ISPI.git
```

## Usage

The paper results can be reproduced by running:
```
python main.py --env halfcheetah-expert-v2 --normalize --alpha=1.0 --aweight=0.7  --seed 2
```

## Paper
[In-Sample Policy Iteration for Offline Reinforcement Learning](https://arxiv.org/pdf/2306.05726.pdf)


## Bibtex
```
@article{hu2023sample,
  title={In-Sample Policy Iteration for Offline Reinforcement Learning},
  author={Hu, Xiaohan and Ma, Yi and Xiao, Chenjun and Zheng, Yan and Meng, Zhaopeng},
  journal={arXiv preprint arXiv:2306.05726},
  year={2023}
}
```

## Result
### Mujoco performance
![image](https://github.com/TJU-DRL-LAB/AI-Optimizer/blob/main/offline-rl-algorithms/ISPI/fig/mujoco-performance.jpg)
### Antmaze performance
![image](https://github.com/TJU-DRL-LAB/AI-Optimizer/blob/main/offline-rl-algorithms/ISPI/fig/antmaze-performance.jpg
)
