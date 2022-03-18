# Dream to Control

Dreamer是论文《[DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION](https://arxiv.org/pdf/1912.01603.pdf)》中设计提出的一个强化学习智能体，通过潜在的想象力从图片中学习长期的行为。在视觉控制任务中和现有方法相比具有良好的效果。这里提供的代码为dreamer的实现。

如需引用需添加：

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```



### Introduction

<img src="/Users/lihao/Library/Application Support/typora-user-images/截屏2021-12-07 下午10.56.44.png" alt="截屏2021-12-07 下午10.56.44" style="zoom:50%;" />

由上图可以看出Dreamer由三个部分组成:

 (a) 从过去的经验数据集学习，agent学习将观察和动作编码为紧凑的潜在状态 (绿点)，例如通过重建，并预测环境奖励 (黄点)。

 (b) 在紧凑的潜在空间中，Dreamer 通过将梯度传播回想象的轨迹来预测状态值 (黄色奖杯) 和动作 (黑红图案)，从而最大化未来值的预测

 (c) agent编码episode的历史以计算当前模型状态并预测要在环境中执行的下一个动作。 

即可以概括为Dreamer 由基于模型的方法的三个经典步骤组成：**学习世界模型**；**从世界模型做出的预测中学习行为**；**在环境中执行学习到的行为来积累新的经验**。这三个步骤可以并行执行，并且在智能体实现目标前一直重复执行。为了考虑超出预测范围的奖励，价值网络评估每个模型状态的未来奖励之和。然后，模型反向传播奖励和价值，以优化行为者网络，从而选择改进的行为，这一过程可以通过下列动图展示：

![111](/Users/lihao/Desktop/111.gif)

实验结果可以在[Project website](https://danijar.com/dreamer)查看。

总体而言，Dreamer得到了较好的结果，在20个视觉控制任务上完成了挑战，这些任务有连续动作和图像作为输入。任务包括平衡、捕捉目标以及各种模拟机器人的移动。与PlaNet（目前最好的基于模型的智能体）、A3C（最好的无模型智能体）、D4PG 等模型进行对比，在 20 个任务上，从性能表现、数据利用效率和计算时间三个方面，Dreamer 都比 D4PG 和 PlaNet 方法优秀。除了在连续控制任务上的实验外，当它被用于离散行为时，Dreamer 也具有良好的泛化能力。此外，文章还进行了一系列消融实验。在达到相同的训练结果的情况下具有更快的训练速度，并且值模型能够使Dreamer鲁棒性更强，在短的视界中表现的效果也很好，同时实验显示实验显示像素重建效果更好，将来可以和Deamer一起应用到复杂任务上。

### Instructions

Get dependencies:

```
pip3 install --user tensorflow-gpu==2.2.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

Train the agent:

```
python3 dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk
```

Generate plots:

```
python3 plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis test/return --bins 3e4
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
```

