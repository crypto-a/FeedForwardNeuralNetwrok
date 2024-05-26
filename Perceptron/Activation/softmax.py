import numpy as np
from .base.baseActivationFunction import BaseActivationFunction


class Softmax(BaseActivationFunction):

    num_classes: int
    output = np.ndarray

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

