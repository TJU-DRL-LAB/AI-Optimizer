# Transfer and Multi-task Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a learning paradigm to solve many decision-making problems, which are usually formalized as Markov Decision Processes (MDP). Recently, Deep Reinforcement Learning (DRL) has achieved a lot of success in human-level control problems, such as video games, robot control, autonomous vehicles, smart grids and so on. However, DRL is still faced with the sample-inefficiency problem especially when the state-action space becomes large, which makes it difficult to learn from scratch. This means the agent has to use a large number of samples to learn a good policy. Furthermore, the sample-inefficiency problem is much more severe in Multiagent Reinforcement Learning (MARL) due to the exponential increase of the state-action space.  

This repository contains the released codes of representative benchmarks and algorithms of TJU-RL-Lab on the topic of Transfer and Multi-task Reinforcement Learning, including the single-agent domain and multi-agent domain, addressing the sample-inefficiency problem in different ways.

This repository will be constantly updated to include new research works.  

<p align="center"><img align="center" src="./assets/overview.png" alt="overview" style="zoom:60%;" /></p>

## Challenges 

**Sample-inefficiency problem**: The main challenge that transfer and multi-task RL aims to sovle is the sample-inefficiency problem. This problem forces the agent to collect a huge amount of training data to learn the optimal policy. For example, the Rainbow DQN requires around 18 million frames of training data to exceed the average level of human players, which is equivalent to 60 hours of games played by human players. However, human players can usually learn an Atari game within a few minutes, and can reach the average level of the same player after one hour of training. 

**Solutions**

- **Transfer RL** which leverages prior knowledge from previously related tasks to accelerate the learning process of RL, has become one popular research direction to significantly improve sample efficiency of DRL. 

- **Multi-task RL**, in which one network learns policies for multiple tasks, has emerged as another promising direction with fast inference and good performance.


## Directory Structure of this Repo

This repository consists of 
 * [Single-agent Transfer RL](#Single-agent Transfer RL)
 * [Single-agent Multi-task RL](#Single-agent Multi-task RL)
 * [Multi-agent Transfer RL](#Multi-agent Transfer RL)

An overview of research works in this repository:

| Category | Sub-Categories | Method |  Is Contained  | Publication | Link |
| ------ | ------ | ----- | --- | ------ | ------ |
| Single-agent Transfer RL | Same-domain Transfer | PTF  | :white_check_mark: |IJCAI 2020| https://dl.acm.org/doi/abs/10.5555/3491440.3491868 |
| Single-agent Transfer RL | Same-domain Transfer | CAPS  | :white_check_mark: |AAMAS 2019| https://dl.acm.org/doi/abs/10.5555/3306127.3331795 |
| Single-agent Transfer RL | Cross-domain Transfer| CAT  | :x: | In progress | N/A |
| Single-agent Transfer RL | Cross-domain Transfer| MIKT  | :x: | UAI 2020 | https://dl.acm.org/doi/abs/10.5555/3306127.3331795 |
| Single-agent Multi-task RL |  | KTM-DRL  | :x: | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/acab0116c354964a558e65bdd07ff047-Abstract.html|
| Multi-agent Transfer RL | Same task, transfer across agents | MAPTF  | :white_check_mark: | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/8d9a6e908ed2b731fb96151d9bb94d49-Abstract.html|
| Multi-agent Transfer RL | Same task, transfer across agents | DVM  | :white_check_mark: | IROS 2019 | https://dl.acm.org/doi/abs/10.5555/3306127.3331795|
| Multi-agent Transfer RL | Policy reuse across tasks | Bayes-ToMoP  | :x: | IJCAI 2019 | https://dl.acm.org/doi/abs/10.5555/3367032.3367121|


## Liscense

This repo uses [MIT Liscense](https://github.com/TJU-DRL-LAB/AI-Optimizer/blob/main/transfer-rl/LICENSE)

## Acknowledgements

**[To add some acknowledgements]**

## *Update Log

2022-03-18:  
-  Repo is created and categories/folders are created.




