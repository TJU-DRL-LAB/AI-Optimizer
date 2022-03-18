# muzero-pytorch
Pytorch Implementation of MuZero : "[Mastering Atari , Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265.pdf)"  based on [pseudo-code](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py) provided by the authors

_Note: This implementation has just been tested on CartPole-v1 and would required modifications(`in config folder`) for other environments_

### Installation
  - Python 3.6, 3.7
  - ```bash
      cd muzero-pytorch
      pip install -r requirements.txt
      ```

### Usage:
* Train: ```python main.py --env CartPole-v1 --case classic_control --opr train --force ```
* Test: ```python main.py --env CartPole-v1 --case classic_control --opr test```
* Visualize results : ```tensorboard --logdir=<result_dir_path>```

|Required Arguments | Description|
|:-------------|:-------------|
| `--env`                          |Name of the environment|
| `--case {atari,classic_control,box2d}` |It's used for switching between different domains(default: None)|
| `--opr {train,test}`             |select the operation to be performed|

|Optional Arguments | Description|
|:-------------|:-------------|
| `--value_loss_coeff`           |Scale for value loss (default: None)|
| `--revisit_policy_search_rate` |Rate at which target policy is re-estimated (default:None)( only valid if ```--use_target_model``` is enabled)|
| `--use_priority`               |Uses priority for  data sampling in replay buffer. Also, priority for new data is calculated based on loss (default: False)|
| `--use_max_priority`           |Forces max priority assignment for new incoming data in replay buffer (only valid if ```--use_priority``` is enabled) (default: False) |
| `--use_target_model`           |Use target model for bootstrap value estimation (default: False)|
| `--result_dir`                 |Directory Path to store results (defaut: current working directory)|
| `--no_cuda`                    |no cuda usage (default: False)|
| `--debug`                      |If enables, logs additional values  (default:False)|
| `--render`                     |Renders the environment (default: False)|
| `--force`                      |Overrides past results (default: False)|
| `--seed`                       |seed (default: 0)|
| `--test_episodes`              |Evaluation episode count (default: 10)|

```Note: default: None => Values are loaded from the corresponding config```

## Training
### CartPole-v1
- Curves represents model evaluation for 5 episodes at 100 step training interval. 
- Also, each curve is a  mean scores over 5 runs (seeds : [0,100,200,300,400])

| |
|:--|
|![](static/imgs/cartpole_test_score.png)|
|![](static/imgs/legend_cartpole.png)|





