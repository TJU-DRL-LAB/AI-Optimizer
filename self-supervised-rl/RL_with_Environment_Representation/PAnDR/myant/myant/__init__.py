from gym.envs.registration import register

register(
    id='myant-v0',
    entry_point='myant.envs:AntEnv',
    max_episode_steps=256,
    reward_threshold=-3.75,
)
