# AAAI2022-Policy-extended Value Function Approximator (PeVFA) and PPO-PeVFA

This is the official implementation (a neat version) of 
our work [What About Inputing Policy in Value Function: Policy Representation and Policy-extended Value Function Approximator](https://arxiv.org/abs/2010.09536)
accepted as oral representation in AAAI 2022.

## Introduction

In this work, we study Policy-extended Value Function Approximator (PeVFA) in Reinforcement Learning (RL), 
which extends conventional value function approximator (VFA) to take as input not only the state (and action) but also an explicit policy representation. 
Such an extension enables PeVFA to **preserve values of multiple policies at the same time** and brings an appealing characteristic, i.e., **value generalization among policies**.

Two typical types of generalization offered by PeVFA are illustrated below:

<div align=center><img align="center" src="./../../../assets/pr_readme_figs/policy_generalization.png" alt="policy_generalization" style="zoom:20%;" /></div>





To make use of value generalization among policies offered by PeVFA, we devise a new form of Generalized Policy Iteraction (GPI), called GPI with PeVFA:

<div align=center><img align="center" src="./../../../assets/pr_readme_figs/GPI_with_PeVFA.png" alt="GPI-with-PeVFA" style="zoom:20%;" /></div>

The key idea is to allow **values learned for historical policies generalize to successive policies along policy improvement path**.

In our experiments, we evaluate the efficacy of value generalization offered by PeVFA and policy representation learning in several OpenAI Gym continuous control tasks. 
For a representative instance of algorithm implementation, Proximal Policy Optimization (PPO) re-implemented under the paradigm of GPI with PeVFA achieves about 40\% performance improvement on its vanilla counterpart in most environments.


## Repo Content
The source code mainly contains:  
-  implementation of our algorithm (PPO-PeVFA) and the vanilla PPO code base;  
-  implementation of policy representation encoding adn learning;
-  the synthetic experiements (TO-ADD).  

All the implementation and experimental details mentioned in our paper and the Supplementary Material can be found in our codes.  

## Installation

Here is an ancient installation guidance which needs step-by-step installation. A more automatic guidance with pip will be considered in the future.


Our codes are implemented with **Python 3.6** and **Tensorflow 1.8**. We recommend the user to install **anaconada** and or **venv** for convenient management of different python envs.

### Environment Setup
We conduct our experiments on [MuJoCo](https://roboti.us/license.html) continuous control tasks in [OpenAI gym](http://gym.openai.com). 
(Now MuJoCo is opensource due to the proposal of DeepMind.)
Please follow the guidance of installation MuJoCo and OpenAI gym as convention.




## Examples  
  
Examples of run commands can be seen in the file below:
> python mujoco_run_ppo_pevf_e2e_ranpr.py

For hyperparameter settings, please refer to our paper for details. Feel free to modify on needs.

## TO-DO
- [ ] Add source code for synthetic experiments


## Citation
If this repository has helped your research, please cite the following:
```
@inproceedings{Tang2021PeVFA,
  author    = {Hongyao Tang and
               Zhaopeng Meng and
               Jianye Hao and
               Chen Chen and
               Daniel Graves and
               Dong Li and
               Changmin Yu and
               Hangyu Mao and
               Wulong Liu and 
               Yaodong Yang and
               Wenyuan Tao and
               Li Wang},
  title     = {What About Inputing Policy in Value Function: Policy Representation and Policy-extended Value Function Approximator},
  booktitle = {Thirty-Sixth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2022},
  pages     = {TBD},
  publisher = {{AAAI} Press},
  year      = {2022},
  url       = {TBD},
  timestamp = {TBD},
  biburl    = {TBD},
  bibsource = {TBD}
}
```
