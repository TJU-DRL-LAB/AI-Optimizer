import logging

from gym import envs
from gym.envs.registration import register


def register_ma_gym_envs():
    logger = logging.getLogger(__name__)

    # Register openai's environments as multi agent
    # This should be done before registering new environments
    env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
    for spec in env_specs:
        register(
            id='ma_' + spec.id,
            entry_point='envs.discrete_magym.envs.openai:MultiAgentWrapper',
            kwargs={'name': spec.id, **spec._kwargs}
        )

    # add new environments : iterate over full observability
    for i, observability in enumerate([False, True]):

        for clock in [False, True]:
            register(
                id='Checkers-v{}'.format(i + (2 if clock else 0)),
                entry_point='envs.discrete_magym.envs.checkers:Checkers',
                kwargs={'full_observable': observability, 'step_cost': -0.01, 'clock': clock}
            )
            register(
                id='Switch2-v{}'.format(i + (2 if clock else 0)),
                entry_point='envs.discrete_magym.envs.switch:Switch',
                kwargs={'n_agents': 2, 'full_observable': observability, 'step_cost': -0.1, 'clock': clock}
            )
            register(
                id='Switch4-v{}'.format(i + (2 if clock else 0)),
                entry_point='envs.discrete_magym.envs.switch:Switch',
                kwargs={'n_agents': 4, 'full_observable': observability, 'step_cost': -0.1, 'clock': clock}
            )

        for num_max_cars in [4, 10]:
            register(
                id='TrafficJunction{}-v'.format(num_max_cars) + str(i),
                entry_point='envs.discrete_magym.envs.traffic_junction:TrafficJunction',
                kwargs={'full_observable': observability, 'n_max': num_max_cars}
            )

        register(
            id='Lumberjacks-v' + str(i),
            entry_point='envs.discrete_magym.envs.lumberjacks:Lumberjacks',
            kwargs={'full_observable': observability}
        )

    register(
        id='Combat-v0',
        entry_point='envs.discrete_magym.envs.combat:Combat',
    )
    register(
        id='PongDuel-v0',
        entry_point='envs.discrete_magym.envs.pong_duel:PongDuel',
    )

    for game_info in [[(5, 5), 2, 1], [(7, 7), 4, 2]]:  # [(grid_shape, predator_n, prey_n),..]
        grid_shape, n_agents, n_preys = game_info
        _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
        register(
            id='{}-v0'.format(_game_name),
            entry_point='envs.discrete_magym.envs.predator_prey:PredatorPrey',
            kwargs={
                'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
            }
        )
        # fully -observable ( each agent sees observation of other agents)
        register(
            id='{}-v1'.format(_game_name),
            entry_point='envs.discrete_magym.envs.predator_prey:PredatorPrey',
            kwargs={
                'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True
            }
        )

        # prey is initialized at random location and thereafter doesn't move
        register(
            id='{}-v2'.format(_game_name),
            entry_point='envs.discrete_magym.envs.predator_prey:PredatorPrey',
            kwargs={
                'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
                'prey_move_probs': [0, 0, 0, 0, 1]
            }
        )

        # full observability + prey is initialized at random location and thereafter doesn't move
        register(
            id='{}-v3'.format(_game_name),
            entry_point='envs.discrete_magym.envs.predator_prey:PredatorPrey',
            kwargs={
                'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
                'prey_move_probs': [0, 0, 0, 0, 1]
            }
        )
