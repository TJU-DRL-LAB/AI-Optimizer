from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """Gaussian kernel with dynamic bandwidth.

    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.

    Args:
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.

    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `tf.Tensor` object of shape (N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A 'tf.Tensor` object of shape (N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.

    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    assert D == D2

    leading_shape = tf.shape(xs)[:-2]

    # Compute the pairwise distances of left and right particles.
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
    # ... x Kx x Ky x D

    if LooseVersion(tf.__version__) <= LooseVersion('1.5.0'):
        dist_sq = tf.reduce_sum(diff**2, axis=-1, keep_dims=False)
    else:
        dist_sq = tf.reduce_sum(diff**2, axis=-1, keepdims=False)
    # ... x Kx x Ky

    # Get median.
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(
        input=tf.reshape(dist_sq, input_shape),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        sorted=True)  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / np.log(Kx)  # ... (shape)
    h = tf.maximum(h, h_min)
    h = tf.stop_gradient(h)  # Just in case.
    h_expanded_twice = tf.expand_dims(tf.expand_dims(h, -1), -1)
    # ... x 1 x 1

    kappa = tf.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # Construct the gradient
    h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
    # ... x 1 x 1 x 1
    kappa_expanded = tf.expand_dims(kappa, -1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return {"output": kappa, "gradient": kappa_grad}
