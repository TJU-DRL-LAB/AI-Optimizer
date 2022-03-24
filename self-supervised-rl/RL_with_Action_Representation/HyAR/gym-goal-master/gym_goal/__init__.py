from gym.envs.registration import register

register(
    id='Goal-v0',
    entry_point='gym_goal.envs:GoalEnv',
)