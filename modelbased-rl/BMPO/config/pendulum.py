params = {
    'type': 'BMPO',
    'universe': 'gym',
    'domain': 'Pendulum',
    'task': 'v0',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 200,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 200,
        'model_retain_epochs': 1,
        'rollout_batch_size': 80e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -1,
        'max_model_t': None,

        'forward_rollout_schedule': [1, 5, 1, 5],
        'backward_rollout_schedule': [1, 5, 1, 5],
        'beta_schedule': [0, 10, 0.01, 0],
        'last_n_epoch': 5,
        'planning_horizon': 6,
        'n_initial_exploration_steps': 200,
    }
}

