# REDQ

## 1. Introduction

Randomized Ensembled Double Q-Learning (REDQ) [1] is a novel model-free algorithm whose sample-efficiency performance is just as good as, if not better than, the state-of-the-art model- based algorithm for the MuJoCo benchmark. REDQ can be used with any standard off-policy model-free algorithm, such as SAC, SOP, TD3, or DDPG. REDQ has the following key components: (i) To improve sample efficiency, the UTD ratio $G$ is much greater than one; (ii) To reduce the variance in the Q-function estimate, REDQ uses an ensemble of ![](https://latex.codecogs.com/svg.latex?N) Q-functions, with each Q-function randomly and independently initialized but updated with the same target; (iii) To reduce over-estimation bias, the target for the Q-function includes a minimization over a random subset ![](https://latex.codecogs.com/svg.latex?M) of the ![](https://latex.codecogs.com/svg.latex?N) Q-functions. The size of the subset ![](https://latex.codecogs.com/svg.latex?M) is kept fixed, and is referred to as the in-target minimization parameter. Since the default choice for ![](https://latex.codecogs.com/svg.latex?M) is ![](https://latex.codecogs.com/svg.latex?M=2), authors refer to the algorithm as Randomized Ensembled Double Q-learning (REDQ). The pseudocode for REDQ is shown in Algorithm 1:

<img src=".\imgs\redq_pseudocode.png" alt="img" style="zoom:80%;" />

## 2. Instruction

```
python redq-train.py --dataset=Hopper-v2 --seed=0 --gpu=0
```

## 3. Performance

<img src=".\imgs\redq_hopper-v2_performance.png" alt="img" style="zoom: 50%;" />

## Reference

1. Chen X, Wang C, Zhou Z, et al. Randomized ensembled double q-learning: Learning fast without a model[J]. arXiv preprint arXiv:2101.05982, 2021.

   

