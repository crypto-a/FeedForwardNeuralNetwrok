import numpy as np
from .base.baseActivationFunction import BaseActivationFunction


class Softmax(BaseActivationFunction):
    def forward(self, inputs):
        # Ensure inputs are at least 2D for consistency
        inputs = np.atleast_2d(inputs)

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities if inputs.shape[0] > 1 else probabilities.flatten()
        return self.output

    def backward(self, dvalues, output):
        # Ensure dvalues are at least 2D for consistency
        dvalues = np.atleast_2d(dvalues)

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self.dinputs if dvalues.shape[0] > 1 else self.dinputs.flatten()
