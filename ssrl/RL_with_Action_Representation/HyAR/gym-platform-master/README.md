# Platform Domain

![Platform domain initial state](img/platform_domain.png)

The Platform environment [[Masson et al. 2016]](https://arxiv.org/abs/1509.01644) uses a parameterised action space and continuous state space. The task involves an agent learning to avoid enemies and traverse across platforms to reach a goal. Three actions are available to the agent:

- run(dx)
- hop(dx)
- leap(dx)

A dense reward is given to the agent based on the distance it travels. The cumulative return is normalised to 1, achieved by reaching the goal. An episode terminates if the agent touches an enemy or falls into a gap between platforms. 

This code is a port of https://github.com/WarwickMasson/aaai-platformer to use the OpenAI Gym framework.

## Dependencies

- Python 3.5+ (tested with 3.5 and 3.6)
- gym 0.10.5
- pygame 1.9.4
- numpy

## Installation

Install this as any other OpenAI gym environment:

    git clone https://github.com/cycraig/gym-platform
    cd gym-platform
    pip install -e '.[gym-platform]'
    
or 

    pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
    
    
## Example Usage

```python
import gym
import gym_platform
env = gym.make('Platform-v0')
```

See https://github.com/cycraig/MP-DQN for an example on how to make an agent for this environment.
    
## Citing

If you use this domain in your research, please cite the original author:

    @inproceedings{Masson2016ParamActions,
        author = {Masson, Warwick and Ranchod, Pravesh and Konidaris, George},
        title = {Reinforcement Learning with Parameterized Actions},
        booktitle = {Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
        year = {2016},
        location = {Phoenix, Arizona},
        pages = {1934--1940},
        numpages = {7},
        publisher = {AAAI Press},
    }
    
You may also consider citing the following paper:

```bibtex
@article{bester2019mpdqn,
	author    = {Bester, Craig J. and James, Steven D. and Konidaris, George D.},
	title     = {Multi-Pass {Q}-Networks for Deep Reinforcement Learning with Parameterised Action Spaces},
	journal   = {arXiv preprint arXiv:1905.04388},
	year      = {2019},
	archivePrefix = {arXiv},
	eprinttype    = {arxiv},
	eprint    = {1905.04388},
	url       = {http://arxiv.org/abs/1905.04388},
}
```
