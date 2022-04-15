from gym.wrappers.monitoring.stats_recorder import StatsRecorder as SR


class StatsRecorder(SR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_step(self, observation, reward, done, info):
        super().after_step(observation, sum(reward), all(done), info)
