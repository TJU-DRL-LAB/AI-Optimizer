import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        sin1, cos1 = next_obs[:,1], next_obs[:,3]
        sin2, cos2 = next_obs[:,2], next_obs[:,4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        done = y <= 1
        
        done = done[:,None]
        return done