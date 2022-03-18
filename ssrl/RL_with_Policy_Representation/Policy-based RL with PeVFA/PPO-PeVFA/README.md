# Policy-extended Value Function Approximator (PeVFA) and PPO-PeVFA

This is the official implementation (a neat version) of 
our work [What About Inputing Policy in Value Function: Policy Representation and Policy-extended Value Function Approximator](https://arxiv.org/abs/2010.09536)
accepted as oral representation in AAAI 2022.


In this work, we study Policy-extended Value Function Approximator (PeVFA) in Reinforcement Learning (RL), 
which extends conventional value function approximator (VFA) to take as input not only the state (and action) but also an explicit policy representation. 
Such an extension enables PeVFA to preserve values of multiple policies at the same time and brings an appealing characteristic, i.e., value generalization among policies.

Two typical types of generalization of PeVFA are illustrated below:
![policy_generalization](http://rl.beiyang.ren/tju_rl/self-supervised-rl/raw/master/RL%20with%20Policy%20Representation/readme_figs/policy_generalization.png)

To make use of value generalization among policies offered by PeVFA, we devise a new form of Generalized Policy Iteraction (GPI), called GPI with PeVFA:

![GPI-with-PeVFA](http://rl.beiyang.ren/tju_rl/self-supervised-rl/raw/master/RL%20with%20Policy%20Representation/readme_figs/GPI_with_PeVFA.png)

The key idea alues learned for historical policies generalize to successive policies along policy improvement path.

In our experiments, we evaluate the efficacy of value generalization offered by PeVFA and policy representation learning in several OpenAI Gym continuous control tasks. 
For a representative instance of algorithm implementation, Proximal Policy Optimization (PPO) re-implemented under the paradigm of GPI with PeVFA achieves about 40\% performance improvement on its vanilla counterpart in most environments.




## TODO