# Offline RL

 This repo contains a unified opensource code implementation of Offline RL algorithms, which is further developed on the basis of d3rlpy (<https://github.com/takuseno/d3rlpy>). This repo will be constantly updated to include new researches  (The development of this repo is in progress at present.)

## Introduction

Standard reinforcement learning (RL) learns how to perform a task through trial and error, balancing exploration and exploitation to achieve better performance. Offline Reinforcement Learning (Offline RL), also known as Batch Reinforcement Learning (BRL), is a variant of Reinforcement Learning that requires an agent to learn to perform tasks from a fixed dataset without exploration. In other words, Offline RL is a data-driven RL paradigm concerned with learning exclusively from static datasets of previously-collected experiences . In the review paper written by Sergey Levine et al. ["Offline reinforcement learning: Tutorial, review, and perspectives on open problems"](https://arxiv.org/abs/2005.01643), they use the following graph to describe the relationship and difference between Offline RL and standard RL as below:

### Taxonomy of Offline RL

Following the classification method by Aviral Kumar and Sergey Levine in [NeurIPS 2020 Tutorial](https://sites.google.com/view/offlinerltutorial-neurips2020/home), we divide the existing Offline RL algorithms into the following four categories :

- Policy Constraint Methods (PC)
- Value Function Regularization Methods (VR)

- Model-based Methods (MB)

- Uncertainty-based Methods (U)

Besides, we add an additional class of Offline to Online research algorithms:

- Offline to Online (Off2On)

For a tutorial of this taxnomy, we refer the reader to our [ZhiHu blog series](https://zhuanlan.zhihu.com/p/414497708).

### Ecology of Offline RL

We plan to establish the ecology of Offline RL in the future. Driven by three key challenges of Offline RL, we are working on researches to address them respectivly. For the limited data problem in Offline RL, we are working on desgin different data augmentation techniques to expand the orginal datasets. Besides, we are designing the multimodal datasets, which is more in line with the real world.  For the overestimation problem in existing Offline RL methods, we plan to develop a unified algorithmic framework together with a unified opensource code-level implementation framework. Finally, our ultimate goal is to land Offline RL methods in real-world decision-making scenarios by further investigating the offline to online training regime.

![Ecology of Offline RL](https://github.com/TJU-DRL-LAB/AI-Optimizer/blob/main/offline-rl-algorithms/Ecology%20of%20Offline%20RL.png)

## Installation

The algorithms in this repo are all implemented **python 3.5** (and versions above).
**PyTorch** is the main DL code frameworks we adopt in this repo with different choices in different algorithms.

First of all, we recommend the user to install **anaconada** and or **venv** for convenient management of different python envs.

In this repo, the following  environments is needed:

* https://github.com/TJU-DRL-LAB/offline-rl-base

- [OpenAI Gym](https://github.com/openai/gym) (e.g., MuJoCo, Robotics)
- [D4RL](https://github.com/rail-berkeley/d4rl)
- ...



Note that each algorithm may use only one or several environments in the ones listed above. Please refer to the page of specific algorithm for concrete requirements.

To clone this repo:

```
git clone git@github.com:TJU-DRL-LAB/offline-rl-algorithms.git
```

## Example

## An Overall View of Offline RL in This Repo

| Category | Method                                                       | Is Contained | Is ReadME Prepared | Publication | Link |
|----------|--------------------------------------------------------------|--------------|--------------------|-------------|------|
| IL       | BC                                                           | ✅            | ✅                  |             |      |
| PC       | Batch Constrained Q-learning (BCQ)                           | ✅            | ✅                  | ICML 2019   | <https://arxiv.org/pdf/1812.02900.pdf>    |
| PC       | Bootstrapping Error Accumulation Reduction (BEAR)            | ✅            | ✅                  | NIPS 2019  |  <https://proceedings.neurips.cc/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf>    |
| PC       | Advantage-Weighted Regression (AWR)                          | ✅            | ✅                  |   |   <https://arxiv.org/pdf/1910.00177.pdf>   |
| VR       | Conservative Q-Learning (CQL)                                | ✅            | ❌                  | NIPS 2020   |  <https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf>    |
| VR       | Critic Reguralized Regression (CRR)                          | ✅            | ❌                  | NIPS 2020   |  <https://proceedings.neurips.cc//paper/2020/file/588cb956d6bbe67078f29f8de420a13d-Paper.pdf>    |
| VR       | Implicit Q-Learning (IQL)                                    | ✅            | ❌                  | In progress   |  <https://arxiv.org/pdf/2110.06169.pdf>    |
| U        | Uncertainty Weighted Actor Critic (UWAC)                     | ❌            | ❌                  |  ICML 2021           |  <http://proceedings.mlr.press/v139/wu21i/wu21i.pdf>    |
| U        | SAC-N                                                        | ✅            | ❌                  |             |  <https://openreview.net/pdf?id=ZUvaSolQZh3>    |
| U        | Ensemble Diversed Actor Critic (EDAC)                        | ❌           | ❌                  |   NIPS 2021          |  <https://openreview.net/pdf?id=ZUvaSolQZh3>    |
| MB       | Model-based Offline Policy Optimization (MOPO)               | ✅            | ❌                  |    NIPS 2020         |  <https://proceedings.neurips.cc/paper/2020/file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf>    |
| MB       | Conservative Offline Model-Based Policy Optimization (COMBO) | ✅            | ❌                  |   NIPS 2021          |  <https://proceedings.neurips.cc/paper/2021/file/f29a179746902e331572c483c45e5086-Paper.pdf>    |
| Off2On   | Advantage Weighted Actor-Critic (AWAC)                       | ✅            | ✅                  | In progress |  <https://arxiv.org/pdf/2006.09359.pdf>    |
| Off2On   | Balanced Replay (BRED)                                       | ❌          | ❌                  | CoRL 2021   |   <https://arxiv.org/pdf/2107.00591.pdf>   |

## Datastes Provided in Repo

## TODO

- [ ] Update a liscence
- [ ] Update the README files for each branches
- [ ] Check the vadality of codes to release

## Citation

If you use our repo in your work, we ask that you cite our **paper**.

Here is an example BibTeX:

```
@article{aaa22xxxx,
  author    = {tjurllab},
  title     = {A Unified Repo for Offline RL},
  year      = {2022},
  url       = {http://arxiv.org/abs/xxxxxxx},
  archivePrefix = {arXiv}
}
```

## Liscense

**[To change]**

## Acknowledgement

**[To add some acknowledgement]**

## *Update Log

-  
