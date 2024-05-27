from base import BaseActivationFunction

import numpy as np


class ReLU(BaseActivationFunction):
    """
    ReLU activation function implementation

    >>> relu = ReLU()
    >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> relu.forward(x)
    array([0., 0., 0., 1., 2.])

    >>> d_out = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> relu.backward(d_out)
    array([0., 0., 0., 1., 1.])
    """

    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out):
        """
        Backward pass
        :param d_out:
        :param output:
        :return:
        """
        d_input = d_out.copy()
        d_input[self.input <= 0] = 0
        return d_input



