# BCQ

## 1. Introduction

Batch-Constrained deep Q-learning (BCQ) [1] is a batch reinforcement learning method for continuous control. BCQ aims to perform Q-learning while constraining the action space to eliminate actions which are unlikely to be selected by the behavioral policy ![](http://latex.codecogs.com/svg.latex?\\pi_{b}), and are therefore unlikely to be contained in the batch. At its core, BCQ uses a state-conditioned generative model ![](https://latex.codecogs.com/svg.latex?G_{\\omega}:\\mathcal{S}&space;\\rightarrow&space;\\mathcal{A}) to model the distribution of data in the batch, ![](https://latex.codecogs.com/svg.latex?G_{\\omega}&space;\\approx&space;\\pi_{b}) akin to a behavioral cloning model. As it is easier to sample from ![](https://latex.codecogs.com/svg.latex?\\pi_{b}(a&space;\\mid&space;s)) than model ![](https://latex.codecogs.com/svg.latex?\\pi_{b}(a&space;\\mid&space;s)) exactly in a continuous action space, the policy is defined by sampling $N$ actions $a_{i}$ from $G_{\omega}(s)$ and selecting the highest valued action according to a Q-network. Since BCQ was designed for continuous actions, the method also includes a perturbation model $\xi_{\phi}(s, a)$, which is a residual added to the sampled actions in the range $[-\Phi, \Phi]$, and trained with the deterministic policy gradient. Finally the authors include a weighted version of Clipped Double Q-learning to penalize high variance estimates and reduce overestimation bias, using $Q_{\theta}^{k}$ with $k=\{1,2\}$ :
$$
\mathcal{L}(\theta)=\sum_{k}\left(r+\gamma \max _{\hat{a}}\left(\lambda \min _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)+(1-\lambda) \max _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)\right)-Q_{\theta}^{k}(s, a)\right)^{2}
$$
where ![](https://latex.codecogs.com/svg.latex?\\hat{a}=a_{i}&plus;\\xi_{\\phi}\\left(s^{\\prime},&space;a_{i}\\right),&space;\\quad&space;a_{i}&space;\\sim&space;G_{\\omega}\\left(s^{\\prime}\\right).) During evaluation, the policy is defined similarly, by sampling $N$ actions from the generative model, perturbing them and selecting the argmax:

<div align=center><img src="https://latex.codecogs.com/svg.image?\pi(s)=\underset{\hat{a}=a_{i}&plus;\xi_{\phi}\left(s^{\prime},&space;a_{i}\right)}{\operatorname{argmax}}&space;Q_{\theta}^{0}(s,&space;\hat{a}),&space;\quad&space;a_{i}&space;\sim&space;G_{\omega}(s)&space;." title="https://latex.codecogs.com/svg.image?\pi(s)=\underset{\hat{a}=a_{i}+\xi_{\phi}\left(s^{\prime}, a_{i}\right)}{\operatorname{argmax}} Q_{\theta}^{0}(s, \hat{a}), \quad a_{i} \sim G_{\omega}(s) ." /></div>

## 2. Instruction

```
python bcq-train.py --dataset=walker2d-random-v2 --seed=0 --gpu=0
```

## 3. Performance

<img src=".\imgs\5s0ZS1.png" alt="img" style="zoom:80%;" />

## Reference

1. Fujimoto S, Meger D, Precup D. Off-policy deep reinforcement learning without exploration[C]//International Conference on Machine Learning. PMLR, 2019: 2052-2062.

   

