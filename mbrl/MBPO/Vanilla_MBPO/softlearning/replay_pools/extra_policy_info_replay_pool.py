from .simple_replay_pool import SimpleReplayPool


class ExtraPolicyInfoReplayPool(SimpleReplayPool):
    def __init__(self, *args, **kwargs):
        super(ExtraPolicyInfoReplayPool, self).__init__(*args, **kwargs)

        fields = {
            'raw_actions': {
                'shape': self._action_space.shape,
                'dtype': 'float32'
            },
            'log_pis': {
                'shape': (1, ),
                'dtype': 'float32'
            }
        }

        self.add_fields(fields)
