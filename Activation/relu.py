import numpy as np
from .base.baseActivationFunction import BaseActivationFunction


class ReLU(BaseActivationFunction):
    output = np.ndarray

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
