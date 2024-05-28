from base import BaseActivationFunction
import numpy as np


class Softmax(BaseActivationFunction):
    """
    Softmax activation function implementation

    >>> softmax = Softmax()
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> np.allclose(softmax.forward(x), np.array([0.09003057, 0.24472847, 0.66524096]))
    True

    >>> d_out = np.array([1.0, 1.0, 1.0])
    >>> np.allclose(softmax.backward(d_out), np.array([0.0, 0.0, 0.0]))
    True
    """
    output: np.ndarray
    d_inputs: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        :param x: Input array
        :return: Softmax output
        """
        # Handle both 1D and 2D input
        if x.ndim == 1:
            x = x.reshape(1, -1)

        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        return self.output if self.output.shape[0] > 1 else self.output.flatten()

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass
        :param d_out: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        # Handle both 1D and 2D input
        if d_out.ndim == 1:
            d_out = d_out.reshape(1, -1)

        self.d_inputs = np.empty_like(d_out)

        for i, (single_output, single_d_out) in enumerate(zip(self.output, d_out)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.d_inputs[i] = np.dot(jacobian_matrix, single_d_out)

        return self.d_inputs if self.d_inputs.shape[0] > 1 else self.d_inputs.flatten()
