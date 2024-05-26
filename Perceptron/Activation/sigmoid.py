import numpy as np
from .base.baseActivationFunction import BaseActivationFunction

class Sigmoid(BaseActivationFunction):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues, output):
        self.dinputs = dvalues * (output * (1 - output))
        return self.dinputs
