# RL with Environment Representation

Environment Representation is one major category in our taxonomy. 

# To update below

The core research content of environment representation is to capture the **variations of environments** from the distribution of interests.

## Repo Content

This repo contains representative research works of TJU-RL-Lab on the topic of RL with Environment Representation.



## An Overall View of Research Works in This Repo  

This repo will be constantly updated to include new researches made by TJU-RL-Lab. 
(The development of this repo is in progress at present.)

| Method | Is Contained | Is ReadME Prepared | Author | Publication | Link |
| ------ | --- | --- | ------ | ------ | ------ |
| CCM | ❌ | ❌ |Haotian Fu | AAAI 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/16914 |
| PAnDR |❌ | ❌ |Tong Sang| [ICLR 2022 GPL Workshop](https://ai-workshops.github.io/generalizable-policy-learning-in-the-physical-world/) | N/A |


## Installation

The algorithms in this repo are all implemented **python 3.5** (and versions above). **Tensorflow 1.x** and **PyTorch** are the main DL code frameworks we adopt in this repo with different choices in different algorithms.

Note that the algorithms contained in this repo may not use all the same environments. Please check the README of specific algorithms for detailed installation guidance.

## TODO
- [ ] Update README file for PAnDR
- [ ] Tidy up code of PAnDR
- [ ] Upload code of CCM

## Related Work

Here we provide a useful list of representative related works on environment (or task) representation in RL.

### Context Representation of Environments
- Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, Deirdre Quillen. Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML 2019 
- Haotian Fu, Hongyao Tang, Jianye Hao, Chen Chen, Xidong Feng, Dong Li, Wulong Liu. Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning. AAAI 2021
- Kimin Lee, Younggyo Seo, Seunghyun Lee, Honglak Lee, Jinwoo Shin. Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning. ICML 2020
- Bernie Wang, Simon Xu, Kurt Keutzer, Yang Gao, Bichen Wu. Improving Context-Based Meta-Reinforcement Learning with Self-Supervised Trajectory Contrastive Learning. arXiv:2103.06386
- Wenxuan Zhou, Lerrel Pinto, Abhinav Gupta. Environment Probing Interaction Policies. ICLR (Poster) 2019

### Others (Reward Functions, Goals and others)
- Roberta Raileanu, Max Goldstein, Arthur Szlam, Rob Fergus. Fast Adaptation via Policy-Dynamics Value Functions. ICML 2020
- Yujing Hu, Weixun Wang, Hangtian Jia, Yixiang Wang, Yingfeng Chen, Jianye Hao, Feng Wu, Changjie Fan. Learning to Utilize Shaping Rewards: A New Approach of Reward Shaping. NIPS 2020
- Daniel S. Brown, Wonjoon Goo, Prabhat Nagarajan, Scott Niekum. Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations. ICML 2019
- Alexander C. Li, Lerrel Pinto, Pieter Abbeel. Generalized Hindsight for Reinforcement Learning.  NIPS 2020



