# AWAC

## 1.Introduction

AWAC (advantage weighted actor critic) is an algorithm that combines sample-efficient dynamic programming with maximum likelihood policy updates, providing a simple and effective framework that is able to leverage large amounts of offline data and then quickly perform online fine-tuning of RL policies,  in order to reach expert-level performance after collecting a limited amount of interaction data. The full AWAC algorithm for offline RL with online fine-tuning is summarized in Algorithm 1. 

<div align=center><img src=".\img\awac.png" alt="img" style="zoom:80%;" /></div>

In a practical implementation, we can parameterize the actor and the critic by neural networks and perform SGD updates from

<div align=center><img src="https://latex.codecogs.com/svg.image?\theta_{k&plus;1}=\mathop{argmax}\limits_\theta&space;\mathop{\mathbb{E}}_{s,a\sim\beta}[log\pi_{\theta}(a|s)exp(\frac1\lambda&space;A^{\pi_k}(s,a))]" title="https://latex.codecogs.com/svg.image?\theta_{k+1}=\mathop{argmax}\limits_\theta \mathop{\mathbb{E}}_{s,a\sim\beta}[log\pi_{\theta}(a|s)exp(\frac1\lambda A^{\pi_k}(s,a))]" /></div>

and

$$
\phi_k=\mathop{argmin}\limits_\phi \mathbb{E}_D[(Q_\phi(s,a)-y)^2]
$$
AWAC ensures data efficiency with off-policy critic estimation via bootstrapping, and avoids offline bootstrap error with a constrained actor update. By avoiding explicit modeling of the behavior policy, AWAC avoids overly conservative updates. 

## 2. Instruction

```shell
python awac-train.py --dataset=HalfCheetah-v2 --seed=0 --gpu=0
```



## 3.Performance







## Reference

[1] Nair A ,  Dalal M ,  Gupta A , et al. Accelerating Online Reinforcement Learning with Offline Datasets[J].  2020.
