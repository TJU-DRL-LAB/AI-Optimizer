# CQL

## 1. introduction

Conservative Q-learning (CQL) [1] is an algorithmic framework for offline RL that learns a expected lower bound on the policy value, which effectively penalizes the Q function at states in the dataset for actions not observed in the dataset. This enables a conservative estimation of the value function for any policy, mitigating the challenges of over-estimation bias and distribution shift. On d4rl tasks, CQL is implemented on top of soft actor-critic (SAC).The iteration of Q function is shown as 

<div align=center><img src="https://latex.codecogs.com/svg.image?\hat{Q}^{k&plus;1}\leftarrow\min&space;_{Q}&space;\max&space;_{\alpha&space;\geq&space;0}&space;\alpha\left(\mathbb{E}_{\mathbf{s}&space;\sim&space;d^{\pi}&space;\beta(\mathbf{s})}\left[\log&space;\sum_{\mathbf{a}}&space;\exp&space;(Q(\mathbf{s},&space;\mathbf{a}))-\mathbb{E}_{\mathbf{a}&space;\sim&space;\pi_{\beta}(\mathbf{a}&space;\mid&space;\mathbf{s})}[Q(\mathbf{s},&space;\mathbf{a})]\right]-\tau\right)&plus;\frac{1}{2}&space;\mathbb{E}_{\mathbf{s},&space;\mathbf{a},&space;\mathbf{s}^{\prime}&space;\sim&space;\mathcal{D}}\left[\left(Q-\mathcal{B}^{\pi_{k}}&space;\hat{Q}^{k}\right)^{2}\right]" title="https://latex.codecogs.com/svg.image?\hat{Q}^{k+1}\leftarrow\min _{Q} \max _{\alpha \geq 0} \alpha\left(\mathbb{E}_{\mathbf{s} \sim d^{\pi} \beta(\mathbf{s})}\left[\log \sum_{\mathbf{a}} \exp (Q(\mathbf{s}, \mathbf{a}))-\mathbb{E}_{\mathbf{a} \sim \pi_{\beta}(\mathbf{a} \mid \mathbf{s})}[Q(\mathbf{s}, \mathbf{a})]\right]-\tau\right)+\frac{1}{2} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}\left[\left(Q-\mathcal{B}^{\pi_{k}} \hat{Q}^{k}\right)^{2}\right]" /></div>

It has a bootstrap error term and a CQL divergence term, which is a result for the optimal current policy optimized aiming to minimize the Q function(added by an additional term of a Unif regularization over current policy) of actions sampled by the current policy and simultaneously maximize the actions sampled by the behavioral policy,  implicitly shows in the first and second term of the ‘max’ term respectively. ![](https://latex.codecogs.com/svg.latex?\\alpha) is an automatically adjustable value via Lagrangian dual gradient descent and ![](https://latex.codecogs.com/svg.latex?\\tau) is a threshold value. When CQL is running on continuous benchmark like the Mujoco tasks, ![](https://latex.codecogs.com/svg.latex?\\log \\sum_{\\mathbf{a}} \\exp (Q(\\mathbf{s}, \\mathbf{a}))) is computed using importance sampling, shown as follows:

<div align=center><img src="https://latex.codecogs.com/svg.image?\log&space;\sum_{a}&space;\exp&space;Q(s,&space;a)&space;\approx&space;\log&space;\left(\frac{1}{2&space;N}&space;\sum_{a_{i}&space;\sim&space;\operatorname{Unif}(a)}^{N}\left[\frac{\exp&space;Q\left(s,&space;a_{i}\right)}{\operatorname{Unif}(a)}\right]&plus;\frac{1}{2&space;N}&space;\sum_{a_{i}&space;\sim&space;\pi_{\phi}(a&space;\mid&space;s)}^{N}\left[\frac{\exp&space;Q\left(s,&space;a_{i}\right)}{\pi_{\phi}\left(a_{i}&space;\mid&space;s\right)}\right]\right)" title="https://latex.codecogs.com/svg.image?\log \sum_{a} \exp Q(s, a) \approx \log \left(\frac{1}{2 N} \sum_{a_{i} \sim \operatorname{Unif}(a)}^{N}\left[\frac{\exp Q\left(s, a_{i}\right)}{\operatorname{Unif}(a)}\right]+\frac{1}{2 N} \sum_{a_{i} \sim \pi_{\phi}(a \mid s)}^{N}\left[\frac{\exp Q\left(s, a_{i}\right)}{\pi_{\phi}\left(a_{i} \mid s\right)}\right]\right)" /></div>

The policy improvement step is the same as SAC's.  

## 2. Instruction

## 3. Performance

## Reference

1. Kumar A, Zhou A, Tucker G, Levine S. Conservative q-learning for offline reinforcement learning[C]//Advances in Neural Information Processing  Systems. 2020;33:1179-91. 



