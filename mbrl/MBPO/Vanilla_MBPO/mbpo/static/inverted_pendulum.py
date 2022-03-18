import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        notdone = np.isfinite(next_obs).all(axis=-1) \
        		  * (np.abs(next_obs[:,1]) <= .2)
        done = ~notdone

        done = done[:,None]

        return done