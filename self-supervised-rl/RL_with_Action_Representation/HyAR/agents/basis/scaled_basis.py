import numpy as np
from .basis import Basis


class ScaledBasis(Basis):
    """
    Scales variables in the range [0,1]
    """

    def __init__(self, nvars, low, high, bias_unit=False):
        super().__init__(nvars)
        self.low = low
        self.high = high
        self.range = self.high-self.low
        self._bias_unit = bias_unit
        if self._bias_unit:
            self.num_terms += 1

    def scale_state(self, state):
        return (state - self.low)/self.range

    def compute_features(self, state):
        scaled_state = self.scale_state(state)
        if self._bias_unit:
            scaled_state = np.concatenate(([1.], scaled_state))
        return scaled_state
