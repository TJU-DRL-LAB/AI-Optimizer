# BEAR

## 1. Introduction

BEAR-QL. Bootstrapping Error Accumulation Reduction Q-Learning (BEAR) [1] is an actor-critic algorithm which builds on the core idea of BCQ, but instead of using a perturbation model, samples actions from a learned actor. As in BCQ, BEAR trains a generative model of the data distribution in the batch. Using the generative model ![](https://latex.codecogs.com/svg.latex?G_{\\omega}), the actor ![](https://latex.codecogs.com/svg.latex?\\pi_{\\phi}) is trained using the deterministic policy gradient, while minimizing the variance over an ensemble of ![](https://latex.codecogs.com/svg.latex?K) Q-networks, and constraining the maximum mean discrepancy (MMD) between ![](https://latex.codecogs.com/svg.latex?G_{\\omega}) and ![](https://latex.codecogs.com/svg.latex?\\pi_{\\phi}) through dual gradient descent:

<div align=center><img src="https://latex.codecogs.com/svg.image?\mathcal{L}(\phi)=-\left(\frac{1}{K}&space;\sum_{k}&space;Q_{\theta}^{k}(s,&space;\hat{a})-\tau&space;\operatorname{var}_{k}&space;Q_{\theta}^{k}(s,&space;\hat{a})\right)&space;\text&space;{&space;s.t.&space;}&space;\operatorname{MMD}\left(G_{\omega}(s),&space;\pi_{\phi}(s)\right)&space;\leq&space;\epsilon," title="https://latex.codecogs.com/svg.image?\mathcal{L}(\phi)=-\left(\frac{1}{K} \sum_{k} Q_{\theta}^{k}(s, \hat{a})-\tau \operatorname{var}_{k} Q_{\theta}^{k}(s, \hat{a})\right) \text { s.t. } \operatorname{MMD}\left(G_{\omega}(s), \pi_{\phi}(s)\right) \leq \epsilon," /></div>


where ![](https://latex.codecogs.com/svg.latex?\\hat{a}&space;\\sim&space;\\pi_{\\phi}(s)) and the MMD is computed over some choice of kernel. The update rule for the ensemble of Q-networks matches BCQ, except the actions ![](https://latex.codecogs.com/svg.latex?\\hat{a}) can be sampled from the single actor network ![](https://latex.codecogs.com/svg.latex?\\pi_{\\phi}) rather than sampling from a generative model and perturbing:

<div align=center><img src="https://latex.codecogs.com/svg.image?\mathcal{L}(\theta)=\sum_{k}\left(r&plus;\gamma&space;\max&space;_{\hat{a}&space;\sim&space;\pi_{\phi}\left(s^{\prime}\right)}\left(\lambda&space;\min&space;_{k^{\prime}}&space;Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime},&space;\hat{a}\right)&plus;(1-\lambda)&space;\max&space;_{k^{\prime}}&space;Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime},&space;\hat{a}\right)\right)-Q_{\theta}^{k}(s,&space;a)\right)^{2}" title="https://latex.codecogs.com/svg.image?\mathcal{L}(\theta)=\sum_{k}\left(r+\gamma \max _{\hat{a} \sim \pi_{\phi}\left(s^{\prime}\right)}\left(\lambda \min _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)+(1-\lambda) \max _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)\right)-Q_{\theta}^{k}(s, a)\right)^{2}" /></div>

The policy used during evaluation is defined similarly to BCQ, but again samples actions directly from the actor:

<div align=center>
<img src="https://latex.codecogs.com/svg.image?\pi(s)=\underset{\hat{a}}{\operatorname{argmax}}&space;\frac{1}{K}&space;\sum_{k}&space;Q_{\theta}^{k}(s,&space;\hat{a}),&space;\quad&space;\hat{a}&space;\sim&space;\pi_{\phi}(s)." title="https://latex.codecogs.com/svg.image?\pi(s)=\underset{\hat{a}}{\operatorname{argmax}} \frac{1}{K} \sum_{k} Q_{\theta}^{k}(s, \hat{a}), \quad \hat{a} \sim \pi_{\phi}(s)." />
</div>

## 2. Instruction

```
python bear-train.py --dataset=walker2d-random-v2 --seed=0 --gpu=0
```

## 3. Performance

<img src="https://z3.ax1x.com/2021/10/21/5s0eQx.png" alt="img" style="zoom:70%;" />

## Reference

1. Kumar A, Fu J, Soh M, et al. Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction[J]. Advances in Neural Information Processing Systems, 2019, 32: 11784-11794.

   

