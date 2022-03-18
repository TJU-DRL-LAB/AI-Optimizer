# PTF_code

Please refer to https://github.com/tianpeiyang/PTF_code

Source code for paper: Efficient deep reinforcement learning via adaptive policy transfer

For reacher task, requirements follow:https://github.com/martinseilair/dm_control2gym

 * [PTF code](#PTF code)
 * [Installation](#Installation)
 * [Run an experiment](#Run an experiment)
    * [Example](#Example)
    * [Results](#results)
 * [Configuration](#Configuration)
    * [Operating parameters](#Operating parameters)
    * [Core parameters](#Core parameters)
    * [Some experiences setting in paper](#Some experiences setting in paper)
 * [In BibTeX format](#In BibTeX format) 

## PTF code
 * ptf
    * alg (agent polices)
       * A3C
       * PPO
       * DQN
       * PTF_A3C
       * PTF_PPO
       * Caps
    * config (Configuration parameters of each polices and game)
       * a3c_conf
       * ppo_conf
       * ptf_a3c_conf
       * ptf_ppo_conf
       * particle_conf 
       * grid_conf
       * pinball_conf
       * reacher_conf
    * game
       * grid_game (grid make_env)
       * pinball (pinball make_env)
       * control2gym_game (control2gym make_env)
    * run (execute the tasks)
       * run_a3c
       * run_ppo
       * run_dqn
       * run_ptf_a3c
       * run_ptf_ppo
       * run_caps
     * source (opponent policies)
     * util
     * main (entry function)

## Installation
python==3.6.5

pip install -r requirements.txt

## Running Example

#### Example
```
python main.py -a ppo -c ppo_conf -g grid -d grid_conf task=324 
```
some logs will be shown below:
```
win : False,  step : 100,  discounted_reward : 0.0,  discount_reward_mean : 0.0,  undiscounted_reward : 0,  reward_mean : 0.0,  episode : 1,
win : False,  step : 100,  discounted_reward : 0.0,  discount_reward_mean : 0.0,  undiscounted_reward : 0,  reward_mean : 0.0,  episode : 2,
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

Source policies contain pre-trained policies for source tasks. For example, in grid game, 4 pre-trained policies(goals are located at grid 81,459,65,295) are as source policies. Using test mode via `-t` `load_model` can reload the model to render

## Configuration

The config files act as defaults for an algorithm or environment. 

They are all located in `config`.

#### Operating parameters
python main.py -a ppo -c ppo_conf -g grid -d grid_conf task=324 
Take the above example: 
* `-a ppo`: choose an algorithm.
* `-c ppo_conf`: choose corresponding algorithm configuration.
* `-g grid`: game type.
* `-d grid_conf`: game configuration.
* `-t`: evaluation the results, by setting `-t True`, and `-t False` as default.
* `task=324`: the goal is located at 324 position in grid game.
* `option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295]`: load source policy for ptf_a3c and ptf_ppo.

#### Core parameters

Default:
* `option_layer_1=20`
* `option_batch_size=32`
* `c1=0.005`
* `e_greedy_increment=0.005`
* `learning_rate_o=0.00001, learning_rate_t=0.00001`
* `xi=0`

#### Some experiences Example
```
# grid ptf_a3c
python main.py -a ptf_a3c -c ptf_a3c_conf -g grid -d grid_conf -n 20000 -e 99 -s 37 -o adam ENTROPY_BETA=0.0001 n_layer_a_1=100 n_layer_c_1=100 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 reward_decay=0.99 option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295] learning_step=10000 save_per_episodes=1000 save_model=True task=324 c1=0.001 N_WORKERS=8 USE_CPU_COUNT=False
# pinball ptf_a3c
python main.py -a ptf_a3c -c ptf_a3c_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 1 -o adam ENTROPY_BETA=0.0001 n_layer_a_1=100 n_layer_c_1=100 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=3e-4 learning_rate_t=3e-4 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 reward_decay=0.99 option_model_path=['source_policies/a3c/0.90.9/model','source_policies/a3c/0.90.2/model','source_policies/a3c/0.20.9/model'] learning_step=10000 save_per_episodes=1000 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg random_start=True start_position=[[0.6,0.4]] target_position=[0.1,0.1] c1=0.0005 source_policy=a3c save_model=True action_clip=1 reward_normalize=False
# reacher ptf_a3c
python main.py -a ptf_a3c -c ptf_a3c_conf -g reacher -d reacher_conf -n 10000 -e 1000 -s 17 -o adam ENTROPY_BETA=0.0003 n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=1e-4 learning_rate_c=1e-4 learning_rate_o=3e-4 learning_rate_t=3e-4 e_greedy=0.95 e_greedy_increment=1e-2 replace_target_iter=5000 reward_decay=0.99 option_model_path=['source_policies/reacher/t1/model','source_policies/reacher/t2/model','source_policies/reacher/t3/model','source_policies/reacher/t4/model'] learning_step=10000 save_model=True task=hard c1=0.001 source_policy=a3c clip_value=10 batch_size=300 option_batch_size=16 reward_normalize=True done_reward=10 learning_times=1 option_layer_1=20 save_per_episodes=1000
# grid ptf_ppo
python main.py -a ptf_ppo -c ptf_ppo_conf -g grid -d grid_conf -n 20000 -e 99 -s 19 -o adam n_layer_a_1=100 n_layer_c_1=100 c2=0.005 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=3e-4 learning_rate_t=3e-4 reward_decay=0.99 clip_value=0.2 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295] learning_step=10000 save_per_episodes=1000 save_model=True task=324 c3=0.0005
# pinball ptf_ppo
python main.py -a ptf_ppo -c ptf_ppo_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 12345 -o adam n_layer_a_1=256 n_layer_c_1=256 c2=0.0001 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 clip_value=0.2 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=5e-4 replace_target_iter=1000 option_batch_size=16 batch_size=32 reward_decay=0.99 option_model_path=['source_policies/a3c/0.90.9/model','source_policies/a3c/0.90.2/model','source_policies/a3c/0.20.9/model'] learning_step=10000 save_per_episodes=1000 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg start_position=[[0.6,0.4]] random_start=True target_position=[0.1,0.1] c1=0.01 source_policy=a3c save_model=True action_clip=1 reward_normalize=False option_layer_1=32
# Reacher ptf_ppo
python main.py -a ptf_ppo -c ptf_ppo_conf -g reacher -d reacher_conf -n 10000 -e 1000 -s 2 -o adam n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=1e-2 replace_target_iter=1000 reward_decay=0.99 option_model_path=['source_policies/reacher/t1/model','source_policies/reacher/t2/model','source_policies/reacher/t3/model','source_policies/reacher/t4/model'] learning_step=10000 save_per_episodes=1000 task=hard c1=0.001 source_policy=a3c clip_value=10 batch_size=300 option_batch_size=16 reward_normalize=True done_reward=10 option_layer_1=20
```

## In BibTeX format:

```tex
@inproceedings{Yang2020,
  author    = {Tianpei Yang and
               Jianye Hao and
               Zhaopeng Meng and
               Zongzhang Zhang and
               Yujing Hu and
               Yingfeng Chen and
               Changjie Fan and
               Weixun Wang and
               Wulong Liu and
               Zhaodong Wang and
               Jiajie Peng},
  title     = {Efficient Deep Reinforcement Learning via Adaptive Policy Transfer},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence},
  pages     = {3094--3100},
  year      = {2020}
}
```
