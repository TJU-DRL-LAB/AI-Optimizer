from gym.envs.registration import register

register(
    id='Platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
    max_episode_steps=200,
    # TODO: max_episode_steps=200,
    # TODO: reward_threshold=1.0? maybe 0.8 or 0.9
)