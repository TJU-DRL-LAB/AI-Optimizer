# MAPTF

Please refer to https://github.com/tianpeiyang/MAPTF_code

Source code for paper: An Efficient Transfer Learning Framework for Multiagent Reinforcement Learning

 * [MAPTF code](#MAPTF code)
 * [Installation](#Installation)
 * [Run an experiment](#Run an experiment)
    * [Example](#Example)
    * [Results](#results)
 * [Configuration](#Configuration)
    * [Operating parameters](#Operating parameters)
    * [Core parameters](#Core parameters)
    * [Some experiences setting in paper](#Some experiences setting in paper)
 * [In BibTeX format](#In BibTeX format) 

## MAPTF code
 * MAPTF
    * alg (multiagent polices)
       * maddpg
       * muti_ptf_ppo
       * sharing_multi_ppo
       * option
    * config (Configuration parameters of each polices)
       * maddpg_conf (including maddpg and maddpg_sr)
       * ppo_config (including ppo sro shppo and shsro)
       * particle_conf (Configuration of particle game )
       * pacman_conf (Configuration of pacman game)
    * run (execute the tasks)
       * run_maddpg_sr (including maddpg and maddpg_sr)
       * run_multi_ptf_ppo_sro (including ppo sro)
       * run_multi_ptf_shppo_sro (including shppo and shsro)
     * source (opponent policies)
     * util
     * main (entry function)

## Installation
python==3.6.5

pip install -r requirements.txt

## Running Example

#### Example
```
python main.py -a multi_ppo -c ppo_conf -g pacman -d pacman_conf game_name=originalClassic num_adversaries=1 adv_load_model=True adv_load_model_path=source/pacman/original/0/model
```
some logs will be shown below:
```
INFO:tensorflow:Restoring parameters from source/pacman/original/0/model_0.ckpt
win : [False, False, False, False],  step : 100,  discounted_reward : [ 0.61213843 -0.63762798 -0.63762798 -0.63762798],  discount_reward_mean : [ 0.61213843 -0.63762798 -0.63762798 -0.63762798],  undiscounted_reward : [ 0.31 -1.01 -1.01 -1.01],  reward_mean : [ 0.31 -1.01 -1.01 -1.01],  episode : 0,
win : [False, False, False, False],  step : 100,  discounted_reward : [ 0.58945708 -0.63762798 -0.63762798 -0.63762798],  discount_reward_mean : [ 0.60079775 -0.63762798 -0.63762798 -0.63762798],  undiscounted_reward : [ 0.31 -1.01 -1.01 -1.01],  reward_mean : [ 0.31 -1.01 -1.01 -1.01],  episode : 1,
```

#### Results

All results will be stored in the `results/alg_name/game_type/game_name/time` folder, every folder contains `graph`, `log`, `model`, `output`, `args.json`, `command.txt`

If you do not want to save `graph` and `model`, setting param `save_model=False`.
* `graph`: can use `tensorboard --logdir=path` to check the tensorflow graph and loss in terminal.
* `log`: the print results in terminal.
* `model`: models saved every `save_per_episodes` episodes.
* `output.json`: reward results.
* `args.json`: store all params.
* `command.txt`: shell command.

## Source Policy

Source policies contain pre-trained opponent policies. For example, in pac-man, the pac-man agent is the opponent, the policy is a pre-trained PPO; in predator-prey, the blue circle agents are pre-trained using PPO. Using test mode via `-t` `load_model`can reload the model to render

## Configuration

The config files act as defaults for an algorithm or environment. 

They are all located in `config`.

#### Operating parameters

Take the above example: 
* `-a multi_ppo`: choose an algorithm.
* `-c ppo_conf`: choose corresponding algorithm configuration.
* `-g pacman`: game type.
* `-d pacman_conf`: game configuration.
* `-t`: evaluation the results, by setting `-t True`, and `-t False` as default.
* `game_name=originalClassic`: choose a game environment.
* `num_adversaries=1`: as needed.
* `adv_load_model=True adv_load_model_path=source/pacman/original/0/model`: load source policy.
* `adv_use_option, good_use_option`: use option, by setting `True`, `False` as default. Learning ppo, shppo and maddpg, setting `False`, otherwise setting `True` as needed.

#### Core parameters

Default:
* `option_layer_1=128, option_layer_2=128`
* `learning_rate_r=0.0003`
* `embedding_dim=32`
* `option_embedding_layer=64`
* `recon_loss_coef=0.1`
* `option_batch_size=32`
* `c1=0.005`
* `e_greedy_increment=0.001`
* `learning_rate_o=0.00001, learning_rate_t=0.00001`
* `xi=0.005`

#### Some experiences setting in paper
```
#ppo+sro, game type=pacman, game environment=mediumClassic
c1=0.005
```
```
#ppo+sro, game type=pacman, game environment=originalClassic
option_batch_size=128
c1=0.0005
```
```
#maddpg+sro, game type=particle, game environment=simple_tag
option_layer_1=128 option_layer_2=128 
learning_rate_o=0.00001 learning_rate_t=0.00001 
c1=0.005 
xi=0
```
```
#ppo+sro, game type=particle, game environment=simple_tag
option_layer_1=32 option_layer_2=32 
c1=0.1 
option_batch_size=128
```
```
#shsro, game type=particle, game environment=simple_tag
option_layer_1=32 option_layer_2=32 
c1=0.1 
```

MADDPG code follows: https://github.com/openai/maddpg

## In BibTeX format:

```tex
@article{yang2021efficient,
  title={An Efficient Transfer Learning Framework for Multiagent Reinforcement Learning},
  author={Yang, Tianpei and Wang, Weixun and Tang, Hongyao and Hao, Jianye and Meng, Zhaopeng and Mao, Hangyu and Li, Dong and Liu, Wulong and Chen, Yingfeng and Hu, Yujing and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
