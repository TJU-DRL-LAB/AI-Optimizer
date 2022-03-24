# AI-Optimizer
AI-Optimizer is a next-generation deep reinforcement learning suit, providing rich algorithm libraries ranging from model-free to model-based RL algorithms, from single-agent to multi-agent algorithms. Moreover, AI-Optimizer contains a flexible and easy-to-use distributed training framework for efficient policy training.
![](./images/AI_Optimizer_overview.png)

AI-Optimizer now provides the following built-in libraries, and more libraries and implementations are coming soon.
- [Multiagent Reinforcement learning](multiagent-rl)
- [Self-supervised Representation Reinforcement Learning](ssrl)
- [Offline Reinforcement Learning](offline-rl-algorithms)
- [Transfer Reinforcement Learning](transfer-and-multi-task-rl)
- [Model-based reinforcement learning](modelbased-rl)

## Multiagent Reinforcement Learning (MARL)
The Multiagent RL repo contains the released codes of representative research works of TJU-RL-Lab on Multiagent Reinforcement Learning (MARL). The research topics are classified according to the critical challenges of MARL, e.g., the curse of dimensionality (scalability) issue, non-stationarity, multiagent credit assignment, exploration-exploitation tradeoff, and hybrid action. To solve these challenges, we propose a series of algorithms from a different point of view. A big picture is shown below.

<p align="center"><img align="center" src="./multiagent-rl/assets/our-work.png" alt="our solutions"  /></p>



## Offline-rl-algorithms (Offrl)
Offline Reinforcement Learning (Offline RL), also known as Batch Reinforcement Learning (BRL), is a variant of Reinforcement Learning that requires an agent to learn to perform tasks from a fixed dataset without exploration. We plan to establish the ecology of Offline RL in the future. Driven by three critical challenges of Offline RL, we are working on research to address them respectively. For the limited data problem in Offline RL, we are working on designing different data augmentation techniques to expand the original datasets. Besides, we are designing multimodal datasets, which are more in line with the real world. For the overestimation problem in existing Offline RL methods, we plan to develop a unified algorithmic framework and a unified opensource code-level implementation framework. Finally, our ultimate goal is to land Offline RL methods in real-world decision-making scenarios by further investigating the offline to the online training regime.
![Ecology of Offline RL](https://github.com/TJU-DRL-LAB/AI-Optimizer/blob/main/offline-rl-algorithms/Ecology%20of%20Offline%20RL.png)

## Self-supervised Reinforcement Learning (SSRL)
SSRL repo contains the released codes of representative research works of TJU-RL-Lab on Self-supervised Representation Learning for RL. Since the RL agent always receives, processes, and delivers all kinds of data in the learning process (i.e., the typical Agent-Environment Interface), 
how to **properly represent such "data"** is naturally one key point to the effectiveness and efficiency of RL.

In this branch, we focus on three key questions:
- **What should a good representation for RL be?**
- **How can we obtain or realize such good representations?**
- **How can we making use of good representations to improve RL?**

Taking **Self-supervised Learning** (SSL) as our major paradigm for representation learning, we carry out our studies from four perspectives: 
**State Representation**,
**Action Representation**,
**Policy Representation**,
**Environment (and Task) Representation**.

The central contribution of this repo is **A Unified Algorithmic Framework (Implementation Design) of SSRL Algorithm**,
with the ultimate goal of establishing the ecology of SSRL, as illustrated below.

<div align=center><img align="center" src="./self-supervised-rl/assets/Ecology_of_SSRL.png" alt="Ecology of SSRL" style="zoom:40%;" /></div>

See more [here](https://github.com/TJU-DRL-LAB/AI-Optimizer/tree/main/self-supervised-rl).


## Transfer and Multi-task Reinforcement Learning
Recently, Deep Reinforcement Learning (DRL) has achieved a lot of success in human-level control problems, such as video games, robot control, autonomous vehicles, smart grids and so on. However, DRL is still faced with the **sample-inefficiency problem** especially when the state-action space becomes large, which makes it difficult to learn from scratch. This means the agent has to use a large number of samples to learn a good policy. Furthermore, the sample-inefficiency problem is much more severe in Multiagent Reinforcement Learning (MARL) due to the exponential increase of the state-action space.  

**Solutions**

- **Transfer RL** which leverages prior knowledge from previously related tasks to accelerate the learning process of RL, has become one popular research direction to significantly improve sample efficiency of DRL. 

- **Multi-task RL**, in which one network learns policies for multiple tasks, has emerged as another promising direction with fast inference and good performance.

This repository contains the released codes of representative benchmarks and algorithms of TJU-RL-Lab on the topic of Transfer and Multi-task Reinforcement Learning, including the single-agent domain and multi-agent domain, addressing the sample-inefficiency problem in different ways.

<p align="center"><img align="center" src="./images/overview.png" alt="overview" style="zoom:60%;" /></p>


## Model-based Reinforcement Learning (MBRL)
This repo contains a unified opensource code implementation for the Model-Based Reinforcement Learning methods. MBRL-Lib provides implementations of popular MBRL algorithms as examples of using this library. The current classifications of the mainstream algorithms in the modern Model-Based RL area are orthogonal, which means some algorithms can be grouped into different categories according to different perspectives. From the mainstream viewpoint,  we can simply divide `Model-Based RL` into two categories: `How to Learn a Model` and `How to Utilize a Model`.

- `How to Learn a Model` mainly focuses on how to build the environment model. 

- `How to Utilize a Model` cares about how to utilize the learned model. 

 Ignoring the differences in specific methods, the purpose of MBRL algorithms can be more finely divided into four directions as follows: `Reduce Model Error`、`Faster Planning`、` Higher Tolerance to Model Error` 、`Scalability to Harder Problems`.  For the problem of `How to Learn a Model`, we can study on reducing model error to learn a more accurate world model or learning a world model with higher tolerance to model error. For the problem of `How to Utilize a Model`, we can study on faster planning with a learned model or the scalability of the learned model to harder problems.  

![](./images/MBRL_framework.png)Currently, we have implemented Dreamer, MBPO,BMPO, MuZero, PlaNet, SampledMuZero, CaDM and we plan to keep increasing this list in the future.  We will constantly update this repo to include new research made by TJU-DRL-Lab. See more [here](https://github.com/TJU-DRL-LAB/AI-Optimizer/tree/main/modelbased-rl).

# Contributing
AI-Optimizer is still under development. More algorithms and features are going to be added and we always welcome contributions to help make AI-Optimizer better. Feel free to contribute.
