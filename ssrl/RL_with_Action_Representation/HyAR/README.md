# related work

This repository includes several reinforcement learning algorithms for hybrid action space MDPs:
1. HPPO[[Fan et al. 2018]](https://arxiv.org/abs/1903.01344v3)
2. MAHHQN[[Fu et al. 2018]](https://arxiv.org/abs/1903.04959)
3. P-DQN [[Xiong et al. 2018]](https://arxiv.org/abs/1810.06394)
4. PA-DDPG [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143)


## Dependencies

- Python 3.6+ (tested with 3.6 and 3.7)
- pytorch 0.4.1+
- gym 0.10.5
- numpy
- click
- pygame
- numba

## Folder Description
- gym-goal gym-platform and multiagent: Hybrid action environment
- agents：Policy of all algorithms, including pdqn, paddpg, hhqn (benchmark policys) ...; pdqn_MPE, pdqn_MPE_4_direction(random policys)...;
  Note: The difference between all random policys is only in the hybrid action dimension.
- HyAR_RL: HyAR-TD3 (TD3 based) and HyAR-DDPG (DDPG based) algorithms training process.
- Raw_RL: HHQN PDQN PADDPG PATD3 and HPPO algorithms training process

## Domains

Experiment scripts are provided to run each algorithm on the following domains with hybrid actions:

- Platform (https://github.com/cycraig/gym-platform)
- Robot Soccer Goal (https://github.com/cycraig/gym-goal)
- Hard Goal
- Hard Move


## Example Usage

HyAR_RL:
```bash
python main_embedding_platform_td3.py
python main_embedding_platform_ddpg.py
```
Raw_RL:
```bash
python main_platform_td3.py 
python main_platform_ddpg.py
```

## Citing
If this repository has helped your research, please cite the following:
```bibtex
@article{DBLP:journals/corr/abs-2109-05490,
  author    = {Boyan Li and
               Hongyao Tang and
               Yan Zheng and
               Jianye Hao and
               Pengyi Li and
               Zhen Wang and
               Zhaopeng Meng and
               Li Wang},
  title     = {HyAR: Addressing Discrete-Continuous Action Reinforcement Learning
               via Hybrid Action Representation},
  journal   = {CoRR},
  volume    = {abs/2109.05490},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.05490},
  eprinttype = {arXiv},
  eprint    = {2109.05490},
  timestamp = {Tue, 21 Sep 2021 17:46:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2109-05490.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
