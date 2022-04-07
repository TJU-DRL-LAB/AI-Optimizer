# BEAR

## 1. Introduction

BEAR-QL. Bootstrapping Error Accumulation Reduction Q-Learning (BEAR) [1] is an actor-critic algorithm which builds on the core idea of BCQ, but instead of using a perturbation model, samples actions from a learned actor. As in BCQ, BEAR trains a generative model of the data distribution in the batch. Using the generative model $G_{\omega}$, the actor $\pi_{\phi}$ is trained using the deterministic policy gradient, while minimizing the variance over an ensemble of $K$ Q-networks, and constraining the maximum mean discrepancy (MMD) between $G_{\omega}$ and $\pi_{\phi}$ through dual gradient descent:
$$
\mathcal{L}(\phi)=-\left(\frac{1}{K} \sum_{k} Q_{\theta}^{k}(s, \hat{a})-\tau \operatorname{var}_{k} Q_{\theta}^{k}(s, \hat{a})\right) \text { s.t. } \operatorname{MMD}\left(G_{\omega}(s), \pi_{\phi}(s)\right) \leq \epsilon,
$$
where $\hat{a} \sim \pi_{\phi}(s)$ and the MMD is computed over some choice of kernel. The update rule for the ensemble of Q-networks matches BCQ, except the actions $\hat{a}$ can be sampled from the single actor network $\pi_{\phi}$ rather than sampling from a generative model and perturbing:
$$
\mathcal{L}(\theta)=\sum_{k}\left(r+\gamma \max _{\hat{a} \sim \pi_{\phi}\left(s^{\prime}\right)}\left(\lambda \min _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)+(1-\lambda) \max _{k^{\prime}} Q_{\theta^{\prime}}^{k^{\prime}}\left(s^{\prime}, \hat{a}\right)\right)-Q_{\theta}^{k}(s, a)\right)^{2}
$$
The policy used during evaluation is defined similarly to $\mathrm{BCQ}$, but again samples actions directly from the actor:
$$
\pi(s)=\underset{\hat{a}}{\operatorname{argmax}} \frac{1}{K} \sum_{k} Q_{\theta}^{k}(s, \hat{a}), \quad \hat{a} \sim \pi_{\phi}(s) .
$$

## 2. Instruction

```
python bear-train.py --dataset=walker2d-random-v2 --seed=0 --gpu=0
```

## 3. Performance

<img src="https://z3.ax1x.com/2021/10/21/5s0eQx.png" alt="img" style="zoom:70%;" />

## Reference

1. Kumar A, Fu J, Soh M, et al. Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction[J]. Advances in Neural Information Processing Systems, 2019, 32: 11784-11794.

   

