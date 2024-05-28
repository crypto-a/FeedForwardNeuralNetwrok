from base import BaseActivationFunction
import numpy as np


class Sigmoid(BaseActivationFunction):
    """
    Sigmoid activation function implementation

    >>> sigmoid = Sigmoid()
    >>> x = np.array([0.0, 2.0, -2.0])
    >>> np.allclose(sigmoid.forward(x), np.array([0.5, 0.88079708, 0.11920292]))
    True

    >>> d_out = np.array([0.1, 0.2, 0.3])
    >>> backward_output = sigmoid.backward(d_out)
    >>> expected_backward_output = d_out * sigmoid.forward(x) * (1 - sigmoid.forward(x))
    >>> np.allclose(backward_output, expected_backward_output)
    True
    """
    output: np.ndarray
    d_inputs: np.ndarray

    def forward(self, x):
        """
        Forward pass
        :param x: Input array
        :return: Sigmoid output
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, d_out):
        """
        Backward pass
        :param d_out: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        sigmoid_grad = self.output * (1 - self.output)
        self.d_inputs = d_out * sigmoid_grad
        return self.d_inputs
