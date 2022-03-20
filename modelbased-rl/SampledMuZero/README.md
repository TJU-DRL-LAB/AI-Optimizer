# Sampled MuZero

A implementation of **Sampled MuZero** based on the [MuZero-General](https://github.com/werner-duvaud/muzero-general).

MuZero is a state of the art RL algorithm for board games (Chess, Go, ...) and Atari games. [Sampled MuZero](http://arxiv.org/abs/2104.06303) is an extension of the MuZero algorithm that is able to learn in domains with arbitrarily complex action spaces by
planning over sampled actions.

## Features

* [x] complex discrete action spaces
* [ ] Multi-dimension continuous action space
* [ ] Adaptive parameter change
* [ ] Batch MCTS


## Getting started
### Installation

```bash
git clone https://github.com/xxx.git
cd SampledMuZero

pip install -r requirements.txt
```

### Run

```bash
python muzero.py --env cartpole --seed 666 --num_simulations 50 --training_steps 100000
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

### Config

You can adapt the configurations of each game by editing the `MuZeroConfig` class of the respective file in the games folder.


