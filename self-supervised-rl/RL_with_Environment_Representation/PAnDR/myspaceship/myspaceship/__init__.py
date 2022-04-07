from gym.envs.registration import register

register(
    id='spaceship-v0',
    entry_point='myspaceship.envs:SpaceshipEnv',
)