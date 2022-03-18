import itertools
import numpy as np
from numba import jit
from .scaled_basis import ScaledBasis


def _fourier(coefficients, scaled_values):
    return np.cos(np.pi * np.dot(coefficients, scaled_values))


# parallel faster for large arrays, much slower for small ones
@jit(nogil=True, nopython=True, parallel=True)
def _fourier_parallel(coefficients, scaled_values):
    return np.cos(np.pi * np.dot(coefficients, scaled_values))


class FourierBasis(ScaledBasis):
    """
    Fourier basis function approximation. Requires the ranges for each dimension, and is thus able to
    use only sine or cosine (and uses cosine). So, this has half the coefficients that a full Fourier approximation
    would use.

    From the paper:
    G.D. Konidaris, S. Osentoski and P.S. Thomas.
    Value Function Approximation in Reinforcement Learning using the Fourier Basis.
    In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.

    Credit:
    Will Dabney (amarack)
    https://github.com/amarack/python-rl/blob/master/pyrl/basis/fourier.py
    """

    _coefficient_cache = {}
    _shrink_cache = {}

    def __init__(self, nvars, low, high, order=3):
        super().__init__(nvars, low, high, False)
        self.num_terms = int(pow(order + 1.0, nvars))
        self.order = order

        self._coefficients = FourierBasis._coefficient_cache.get((nvars,order))
        if self._coefficients is None:
            it = itertools.product(range(order + 1), repeat=nvars)
            self._coefficients = np.array([list(map(np.float32, x)) for x in it])
            FourierBasis._coefficient_cache[(nvars, order)] = self._coefficients

        self._shrink = FourierBasis._shrink_cache.get((nvars, order))
        if self._shrink is None:
            self._shrink = np.linalg.norm(self._coefficients, axis=1)
            self._shrink[self._shrink == 0.] = 1.
            FourierBasis._shrink_cache[(nvars, order)] = self._shrink

        if self.num_terms > 200000:
            self._fourier_func = _fourier_parallel
        else:
            self._fourier_func = _fourier

    def compute_features(self, state):
        """
        Computes the Fourier basis features for the given state

        :param state: state variables (scaled in [0,1])
        :return:
        """
        scaled_state = super().compute_features(state)
        return self._fourier_func(self._coefficients, scaled_state)
        # return np.cos(np.pi * np.dot(self._coefficients, scaled_state))

    def __str__(self):
        return "FourierBasis (o{0:d}) {1:d} terms".format(self.order, self.num_terms)
