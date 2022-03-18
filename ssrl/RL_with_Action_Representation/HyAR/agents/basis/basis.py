import numpy as np


class Basis(object):
    """
    Trivial basis
    """

    def __init__(self, nvars):
        self.num_terms = nvars
        self._shrink = np.ones((self.num_terms,))

    def get_num_basis_functions(self):
        return self.num_terms

    def compute_features(self, state):
        return state

    def get_shrink(self):
        return self._shrink
