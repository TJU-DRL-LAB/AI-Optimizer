import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        z = next_obs[:,0]
        done = (z < 1.0) + (z > 2.0)

        done = done[:,None]
        return done