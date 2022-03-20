# Deep Planning Network

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson

![PlaNet policies and predictions](https://imgur.com/UeeQIfo.gif)

This project provides the open source implementation of the PlaNet agent
introduced in [Learning Latent Dynamics for Planning from Pixels][paper].
PlaNet is a purely model-based reinforcement learning algorithm that solves
control tasks from images by efficient planning in a learned latent space.
PlaNet competes with top model-free methods in terms of final performance and
training time while using substantially less interaction with the environment.

If you find this open source release useful, please reference in your paper:

```
@inproceedings{hafner2019planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle={International Conference on Machine Learning},
  pages={2555--2565},
  year={2019}
}
```

## Method

![PlaNet model diagram](https://i.imgur.com/fpvrAqw.png)

PlaNet models the world as a compact sequence of hidden states. For planning,
we first encode the history of past images into the current state. From there,
we efficiently predict future rewards for multiple action sequences in latent
space. We execute the first action of the best sequence found and replan after
observing the next image.

Find more information:

- [Google AI Blog post][blog]
- [Project website][website]
- [PDF paper][paper]

[blog]: https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html
[website]: https://danijar.com/project/planet/
[paper]: https://arxiv.org/pdf/1811.04551.pdf

## Instructions

To train an agent, install the dependencies and then run:

```sh
python3 -m planet.scripts.train --logdir /path/to/logdir --params '{tasks: [cheetah_run]}'
```

The code prints `nan` as the score for iterations during which no summaries
were computed.

The available tasks are listed in `scripts/tasks.py`. The default parameters
can be found in `scripts/configs.py`. To run the experiments from our
paper, pass the following parameters to `--params {...}` in addition to the
list of tasks:

| Experiment | Parameters |
| :--------- | :--------- |
| PlaNet | No additional parameters. |
| Random data collection | `planner_iterations: 0, train_action_noise: 1.0` |
| Purely deterministic | `mean_only: True, divergence_scale: 0.0` |
| Purely stochastic | `model: ssm` |
| One agent all tasks | `collect_every: 30000` |

Please note that the agent has seen some improvements so the results may be a
bit different now.

## Modifications

These are good places to start when modifying the code:

| Directory | Description |
| :-------- | :---------- |
| `scripts/configs.py` | Add new parameters or change defaults. |
| `scripts/tasks.py` | Add or modify environments. |
| `models` | Add or modify latent transition models. |
| `networks` | Add or modify encoder and  decoder networks. |

Tips for development:

- You can set `--config debug` to reduce the episode length, batch size, and
  collect data more freqnently. This helps to quickly reach all parts of the
  code.
- You can use `--num_runs 1000 --resume_runs False` to automatically start new
  runs in sub directories of the logdir every time to execute the script.
- Environments live in separate processes by default. Some environments work
  better when separated into threads instead by specifying `--params
  '{isolate_envs: thread}'`.

## Dependencies

The code was tested under Ubuntu 18 and uses these packages:

- tensorflow-gpu==1.13.1
- tensorflow_probability==0.6.0
- dm_control (`egl` [rendering option][dmc-rendering] recommended)
- gym
- scikit-image
- scipy
- ruamel.yaml
- matplotlib

[dmc-rendering]: https://github.com/deepmind/dm_control#rendering

Disclaimer: This is not an official Google product.
