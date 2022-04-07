# RL with State Representation

State Representation is one major category in our taxonomy. 
The core research content of state representation is to learn the abstraction/representation of original state space, which is usually high-dimensional, complex and difficult to deal with directly by conventional RL algorithms.
The purpose of state representation learning is to make RL effective, efficient and optimal in especially practical decision-making problems.

In essence, to learn state representation is the process of discovering the decision-related or decision-agnositic (depending on the specific purposes) images of original MDPs.

In our opinion, RL with State Representation contains the research on:
- **What an optimal state representation should be like. (Theories on Abstraction and Generalization)**
- **How to obtain or learn desired state representation in specific cases. (Methods of Learning State Representation)**
- **How to deal with the co-learning and inter-dependence between state representation and RL policy/value functions. (Studies on Learning Dynamics)**

### Two-Timescale Model of RL with State Representation

The conventional paradigm of RL with State Representation can be demonstrated by a _two-timescale model_, which is illustrated below (modified from [Chung et al., ICLR 2019](https://openreview.net/forum?id=rJleN20qK7)):

<div align=center><img align="center" src="./../assets/sr_readme_figs/sr_framework.png" alt="state_representation_framework" style="zoom:20%;" /></div>

We may note the three features in the paradigm: 1) the state representation, denoted by 𝑥_𝜃(𝑠), is produced by a learnable mapping usually implemented by neural networks; 2) the state representation is learned by optimizing an auxilliary task, denoted by the surrogate 𝑌 ̂(𝑠); 3) finally, the state representation is taken as input by RL functions, e.g., 𝑉 ̂(𝑠), and involved in conventional RL process.

The word 'Two-Timescale' means that state representation and RL functions are often learned at two timescales (or even two stages as in pre-trained state representation).




## Related Work

Here we provide a useful list of representative related works on state representation in RL.

### Abstraction, Optimality and Generalization

- Lihong Li, Thomas J. Walsh, Michael L. Littman. Towards a Unified Theory of State Abstraction for MDPs. ISAIM 2006
- David Abel, D. Ellis Hershkowitz, Michael L. Littman. Near Optimal Behavior via Approximate State Abstraction. ICML 2016
- Charline Le Lan, Marc G. Bellemare, Pablo Samuel Castro. Metrics and Continuity in Reinforcement Learning. AAAI 2021
- Clare Lyle, Mark Rowland, Georg Ostrovski, Will Dabney. On the Effect of Auxiliary Tasks on Representation Dynamics. AISTATS 2021
- David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup, Michael L. Littman. Value Preserving State-Action Abstractions. AISTATS 2020



### Representations Developed from General Un-/Self-supervised Learning Principles

- Michael Laskin, Aravind Srinivas, Pieter Abbeel. CURL: Contrastive Unsupervised Representations for Reinforcement Learning. ICML 2020
- Ilya Kostrikov, Denis Yarats, Rob Fergus. Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels.  arXiv:2004.13649
- Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, Aravind Srinivas. Reinforcement Learning with Augmented Data. NeurIPS 2020
- Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning. arXiv:2107.09645
- Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. Reinforcement Learning with Prototypical Representations.  ICML 2021.

### Representations Built on RL Elements

- Max Schwarzer, Ankesh Anand, Rishab Goel, R. Devon Hjelm, Aaron C. Courville, Philip Bachman. Data-Efficient Reinforcement Learning with Momentum/Self- Predictive Representations. arXiv:2007.05929
- Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, Sergey Levine. Learning Invariant Representations for Reinforcement Learning without Reconstruction. ICLR 2021
- Guoqing Liu, Chuheng Zhang, Li Zhao, Tao Qin, Jinhua Zhu, Jian Li, Nenghai Yu, Tie-Yan Liu. Return-Based Contrastive Representation Learning for Reinforcement Learning. ICLR 2021
- Dibya Ghosh, Abhishek Gupta, Sergey Levine. Learning Actionable Representations with Goal Conditioned Policies. ICLR (Poster) 2019
- Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, Marc G. Bellemare. DeepMDP: Learning Continuous Latent Space Models for Representation Learning. ICML 2019
- Rishabh Agarwal, Marlos C. Machado, Pablo Samuel Castro, Marc G. Bellemare. Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning. ICLR 2021

### Representations for Model-based RL

- Danijar Hafner, Timothy P. Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Learning Latent Dynamics for Planning from Pixels. ICML 2019
- Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020
- Changmin Yu, Dong Li, Hangyu Mao, Jianye Hao, Neil Burgess. Learning State Representations via Temporal Cycle-Consistency Constraint in Model-Based Reinforcement Learning. ICLR 2021 Workshop on SSL-RL
- Thomas N. Kipf, Elise van der Pol, Max Welling. Contrastive Learning of Structured World Models. ICLR 2020

### Decoupled State Representation Learning

- Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. Reinforcement Learning with Prototypical Representations. ICML 2021.
- Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin. Decoupling Representation Learning from Reinforcement Learning. ICML 2021
- Hao Liu, Pieter Abbeel. Unsupervised Active Pre-Training for Reinforcement Learning. ICLR 2021 (rejected).
- Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, R. Devon Hjelm, Philip Bachman, Aaron C. Courville. Pretraining Representations for Data-Efficient Reinforcement Learning. arXiv.2106.04799
- Hao Liu, Pieter Abbeel. APS: Active Pretraining with Successor Features. ICML 2021

### Effects of Network and Architecture on State Representation Learning

- Kei Ota, Tomoaki Oiki, Devesh K. Jha, Toshisada Mariyama, Daniel Nikovski. Can Increasing Input Dimensionality Improve Deep Reinforcement Learning? ICML 2020
- Samarth Sinha, Homanga Bharadhwaj, Aravind Srinivas, Animesh Garg. D2RL: Deep Dense Architectures in Reinforcement Learning. arXiv:2010.09163
- Kei Ota, Devesh K. Jha, Asako Kanezaki. Training Larger Networks for Deep Reinforcement Learning.  arXiv:2102.07920

### Actor-critic State Representation Interference

- Karl Cobbe, Jacob Hilton, Oleg Klimov, John Schulman. Phasic Policy Gradient. ICML 2021
- Roberta Raileanu, Rob Fergus. Decoupling Value and Policy for Generalization in Reinforcement Learning. ICML 2021

### Sparse State Representation

- Vincent Liu, Raksha Kumaraswamy, Lei Le, Martha White. The Utility of Sparse Representations for Control in Reinforcement Learning. AAAI 2019.
- Yangchen Pan, Kirby Banman, Martha White. Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online. ICLR 2021

### Bias & Issues of State Representation Learning

- Tyler Lu, Dale Schuurmans, Craig Boutilier. Non-delusional Q-learning and value-iteration. NeurIPS 2018.
- Dijia Su, Jayden Ooi, Tyler Lu, Dale Schuurmans, Craig Boutilier. ConQUR: Mitigating Delusional Bias in Deep Q-Learning. ICML 2020.
- Sina Ghiassian, Banafsheh Rafiee, Yat Long Lo, Adam White. Improving Performance in Reinforcement Learning by Breaking Generalization in Neural Networks. AAMAS 2020.
- Joshua Achiam, Ethan Knight, Pieter Abbeel. Towards Characterizing Divergence in Deep Q-Learning. arXiv:1903.08894, 2019.

### Deeper Looks of Auxiliary Tasks for State Representation Learning (Effects of Representation Learning Dynamics)

- Wesley Chung, Somjit Nath, Ajin Joseph, Martha White. Two-Timescale Networks for Nonlinear Value Function Approximation. ICLR (Poster) 2019
- Will Dabney, André Barreto, Mark Rowland, Robert Dadashi, John Quan, Marc G. Bellemare, David Silver. The Value-Improvement Path: Towards Better Representations for Reinforcement Learning. AAAI 2021
- Robert Dadashi, Marc G. Bellemare, Adrien Ali Taïga, Nicolas Le Roux, Dale Schuurmans. The Value Function Polytope in Reinforcement Learning. ICML 2019 
- Marc G. Bellemare, Will Dabney, Robert Dadashi, Adrien Ali Taïga, Pablo Samuel Castro, Nicolas Le Roux, Dale Schuurmans, Tor Lattimore, Clare Lyle. A Geometric Perspective on Optimal Representations for Reinforcement Learning. NeurIPS 2019
- David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup, Michael L. Littman. Value Preserving State-Action Abstractions. AISTATS 2020
- Clare Lyle, Mark Rowland, Georg Ostrovski, Will Dabney. On The Effect of Auxiliary Tasks on Representation Dynamics. arXiv:2102.13089
- Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine. Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning. arXiv:2010.14498












