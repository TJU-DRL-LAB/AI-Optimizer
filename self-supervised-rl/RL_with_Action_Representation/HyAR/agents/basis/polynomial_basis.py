import numpy as np
from .basis import Basis


class PolynomialBasis(Basis):
    """
    Induces a polynomial basis (excluding combinations) over the state variables.

    Example:
    --------
    basis = PolynomialBasis(2, order=2)
    basis.compute_features(np.array([-2, 3])
        array([ -2, 3, 4, 9 ])
    """

    def __init__(self, nvars, order=2, bias_unit=False):
        super().__init__(nvars)
        self.order = order
        self._bias_unit = bias_unit
        self.num_terms = order*nvars
        if self._bias_unit:
            self.num_terms += 1

    def compute_features(self, state):
        features = np.concatenate([state**i for i in range(1,self.order+1)])
        if self._bias_unit:
            features = np.concatenate(([1.], features))
        return features
