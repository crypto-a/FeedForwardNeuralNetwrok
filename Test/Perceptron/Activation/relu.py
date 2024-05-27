import numpy as np
from .base.baseActivationFunction import BaseActivationFunction


class ReLU(BaseActivationFunction):
    output = np.ndarray
    inputs = np.ndarray

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues, output):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
