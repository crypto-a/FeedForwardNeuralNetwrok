from Base import BaseActivationFunction
import numpy as np

class Sigmoid(BaseActivationFunction):
    """
    Sigmoid activation function implementation
    """

    output: np.ndarray
    d_inputs: np.ndarray

    def forward(self, x):
        """
        Forward pass
        :param x: Input array
        :return: Sigmoid output
        """
        # Use a numerically stable sigmoid implementation
        self.output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
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
