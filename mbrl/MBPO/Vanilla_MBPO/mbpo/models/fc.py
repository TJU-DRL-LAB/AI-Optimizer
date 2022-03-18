from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

import pdb


class FC:
    """Represents a fully-connected layer in a network.
    """
    _activations = {
        None: tf.identity,
        "ReLU": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self, output_dim, input_dim=None,
                 activation=None, weight_decay=None, ensemble_size=1):
        """Initializes a fully connected layer.

        Arguments:
            output_dim: (int) The dimensionality of the output of this layer.
            input_dim: (int/None) The dimensionality of the input of this layer.
            activation: (str/None) The activation function applied on the outputs.
                                    See FC._activations to see the list of allowed strings.
                                    None applies the identity function.
            weight_decay: (float) The rate of weight decay applied to the weights of this layer.
            ensemble_size: (int) The number of networks in the ensemble within which this layer will be used.
        """
        # Set layer parameters
        self.input_dim, self.output_dim = input_dim, output_dim
        self.activation = activation
        self.weight_decay = weight_decay
        self.ensemble_size = ensemble_size

        # Initialize internal state
        self.variables_constructed = False
        self.weights, self.biases = None, None
        self.decays = None

    def __repr__(self):
        return "FC(output_dim={!r}, input_dim={!r}, activation={!r}, weight_decay={!r}, ensemble_size={!r})"\
            .format(
                self.output_dim, self.input_dim, self.activation, self.weight_decay, self.ensemble_size
            )

    #### Extensions
    def get_model_vars(self, idx, sess):
        weights, biases = sess.run([self.weights, self.biases])
        weight = weights[idx].copy()
        bias = biases[idx].copy()
        return {'weights': weight, 'biases': bias}

    def set_model_vars(self, variables):
        ops = [getattr(self, attr).assign(var) for attr, var in variables.items()]
        return ops

    def reset(self, sess):
        sess.run(self.weights.initializer)
        sess.run(self.biases.initializer)

    #######################
    # Basic Functionality #
    #######################

    def compute_output_tensor(self, input_tensor):
        """Returns the resulting tensor when all operations of this layer are applied to input_tensor.

        If input_tensor is 2D, this method returns a 3D tensor representing the output of each
        layer in the ensemble on the input_tensor. Otherwise, if the input_tensor is 3D, the output
        is also 3D, where output[i] = layer_ensemble[i](input[i]).

        Arguments:
            input_tensor: (tf.Tensor) The input to the layer.

        Returns: The output of the layer, as described above.
        """
        # Get raw layer outputs
        if len(input_tensor.shape) == 2:
            raw_output = tf.einsum("ij,ajk->aik", input_tensor, self.weights) + self.biases
        elif len(input_tensor.shape) == 3 and input_tensor.shape[0].value == self.ensemble_size:
            raw_output = tf.matmul(input_tensor, self.weights) + self.biases
        else:
            raise ValueError("Invalid input dimension.")

        # Apply activations if necessary
        return FC._activations[self.activation](raw_output)

    def get_decays(self):
        """Returns the list of losses corresponding to the weight decay imposed on each weight of the
        network.

        Returns: the list of weight decay losses.
        """
        return self.decays

    def copy(self, sess=None):
        """Returns a Layer object with the same parameters as this layer.

        Arguments:
            sess: (tf.Session/None) session containing the current values of the variables to be copied.
                  Must be passed in to copy values.
            copy_vals: (bool) Indicates whether variable values will be copied over.
                       Ignored if the variables of this layer has not yet been constructed.

        Returns: The copied layer.
        """
        new_layer = eval(repr(self))
        return new_layer

    #########################################################
    # Methods for controlling internal Tensorflow variables #
    #########################################################

    def construct_vars(self):
        """Constructs the variables of this fully-connected layer.

        Returns: None
        """
        if self.variables_constructed:  # Ignore calls to this function once variables are constructed.
            return
        if self.input_dim is None or self.output_dim is None:
            raise RuntimeError("Cannot construct variables without fully specifying input and output dimensions.")

        # Construct variables
        self.weights = tf.get_variable(
            "FC_weights",
            shape=[self.ensemble_size, self.input_dim, self.output_dim],
            initializer=tf.truncated_normal_initializer(stddev=1/(2*np.sqrt(self.input_dim)))
        )
        self.biases = tf.get_variable(
            "FC_biases",
            shape=[self.ensemble_size, 1, self.output_dim],
            initializer=tf.constant_initializer(0.0)
        )

        if self.weight_decay is not None:
            self.decays = [tf.multiply(self.weight_decay, tf.nn.l2_loss(self.weights), name="weight_decay")]
        self.variables_constructed = True

    def get_vars(self):
        """Returns the variables of this layer.
        """
        return [self.weights, self.biases]

    ########################################
    # Methods for setting layer parameters #
    ########################################

    def get_input_dim(self):
        """Returns the dimension of the input.

        Returns: The dimension of the input
        """
        return self.input_dim

    def set_input_dim(self, input_dim):
        """Sets the dimension of the input.

        Arguments:
            input_dim: (int) The dimension of the input.

        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.input_dim = input_dim

    def get_output_dim(self):
        """Returns the dimension of the output.

        Returns: The dimension of the output.
        """
        return self.output_dim

    def set_output_dim(self, output_dim):
        """Sets the dimension of the output.

        Arguments:
            output_dim: (int) The dimension of the output.

        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.output_dim = output_dim

    def get_activation(self, as_func=True):
        """Returns the current activation function for this layer.

        Arguments:
            as_func: (bool) Determines whether the returned value is the string corresponding
                     to the activation function or the activation function itself.

        Returns: The activation function (string/function, see as_func argument for details).
        """
        if as_func:
            return FC._activations[self.activation]
        else:
            return self.activation

    def set_activation(self, activation):
        """Sets the activation function for this layer.

        Arguments:
            activation: (str) The activation function to be used.

        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.activation = activation

    def unset_activation(self):
        """Removes the currently set activation function for this layer.

        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.set_activation(None)

    def get_weight_decay(self):
        """Returns the current rate of weight decay set for this layer.

        Returns: The weight decay rate.
        """
        return self.weight_decay

    def set_weight_decay(self, weight_decay):
        """Sets the current weight decay rate for this layer.

        Returns: None
        """
        self.weight_decay = weight_decay
        if self.variables_constructed:
            if self.weight_decay is not None:
                self.decays = [tf.multiply(self.weight_decay, tf.nn.l2_loss(self.weights), name="weight_decay")]

    def unset_weight_decay(self):
        """Removes weight decay from this layer.

        Returns: None
        """
        self.set_weight_decay(None)
        if self.variables_constructed:
            self.decays = []

    def set_ensemble_size(self, ensemble_size):
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.ensemble_size = ensemble_size

    def get_ensemble_size(self):
        return self.ensemble_size
