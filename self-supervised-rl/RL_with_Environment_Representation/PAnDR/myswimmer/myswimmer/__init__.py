from gym.envs.registration import register

register(
    id='myswimmer-v0',
    entry_point='myswimmer.envs:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)
