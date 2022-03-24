import numpy as np
from .basis import Basis


class SimpleBasis(Basis):
    """
    Simple basis with an optional bias unit
    """

    def __init__(self, nvars, bias_unit=False):
        super().__init__(nvars)
        self._bias_unit = bias_unit
        if self._bias_unit:
            self.num_terms += 1

    def compute_features(self, state):
        if self._bias_unit:
            state = np.concatenate(([1.], state))
        return state
