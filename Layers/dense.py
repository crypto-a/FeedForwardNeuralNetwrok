from Base import BaseLayer
from Base import BaseActivationFunction
from Activation import *
import numpy as np


class Dense(BaseLayer):
    """
    A dense layer is a layer that is fully connected to the previous layer.\
    The output of the layer is computed as follows:
    output = inputs @ weights + biases

    >>> layer = Dense(3, 2, 'relu')
    >>> layer.weights.shape
    (3, 2)
    >>> layer.bias.shape
    (1, 2)
    >>> isinstance(layer.activation, ReLU)
    True
    """

    weights: np.ndarray
    bias: np.ndarray

    activation: BaseActivationFunction

    inputs: np.ndarray
    output_preactivation: np.ndarray
    output: np.ndarray

    d_weights: np.ndarray
    d_bias: np.ndarray

    def __init__(self, n_inputs: int, n_neurons: int, activation: str):
        """
        constructor for the dense layer class

        :param n_inputs:
        :param n_neurons:
        :param activation:
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        activation_cases = {
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'softmax': Softmax
        }

        if activation not in activation_cases:
            raise ValueError(f"Activation function {activation} is not available")

        self.activation = activation_cases[activation]()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        forward pass for the dense layer

        :param inputs:
        :return:

        >>> layer = Dense(3, 2, 'relu')
        >>> inputs = np.array([[1.0, 2.0, 3.0]])
        >>> output = layer.forward(inputs)
        >>> output.shape
        (1, 2)
        """
        self.inputs = inputs
        self.output_preactivation = np.dot(inputs, self.weights) + self.bias

        self.output = self.activation.forward(self.output_preactivation)
        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        backward pass for the dense layer

        :param d_output:
        :return:

        >>> layer = Dense(3, 2, 'relu')
        >>> inputs = np.array([[1.0, 2.0, 3.0]])
        >>> output = layer.forward(inputs)
        >>> d_output = np.array([[0.1, 0.2]])
        >>> d_inputs = layer.backward(d_output)
        >>> output.shape
        (1, 2)
        >>> d_inputs.shape
        (1, 3)
        >>> layer.d_weights.shape
        (3, 2)
        >>> layer.d_bias.shape
        (1, 2)
        """
        # Calculate gradients
        d_preactivation = self.activation.backward(d_output)
        d_inputs = np.dot(d_preactivation, self.weights.T)
        self.d_weights = np.dot(self.inputs.T, d_preactivation)
        self.d_bias = np.sum(d_preactivation, axis=0, keepdims=True)

        return d_inputs

    def update_params(self, learning_rate: float):
        """
        Update the weights and biases using the computed gradients.

        :param learning_rate: Learning rate for the update.

        >>> layer = Dense(3, 2, 'relu')
        >>> inputs = np.array([[1.0, 2.0, 3.0]])
        >>> forward_output = layer.forward(inputs)
        >>> d_output = np.array([[0.1, 0.2]])
        >>> backward_output = layer.backward(d_output)
        >>> old_weights = layer.weights.copy()
        >>> old_bias = layer.bias.copy()
        >>> layer.update_params(0.1)
        >>> forward_output.shape
        (1, 2)
        >>> backward_output.shape
        (1, 3)
        >>> np.allclose(layer.weights, old_weights - 0.1 * layer.d_weights)
        True
        >>> np.allclose(layer.bias, old_bias - 0.1 * layer.d_bias)
        True
        """
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)



