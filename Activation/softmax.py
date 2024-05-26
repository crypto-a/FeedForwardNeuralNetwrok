import numpy as np
from .base.baseActivationFunction import BaseActivationFunction


class Softmax(BaseActivationFunction):
    output = np.ndarray

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities