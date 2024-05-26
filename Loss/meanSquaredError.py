import numpy as np
from .base.baseLossFunction import BaseLossFunction


class MeanSquaredError(BaseLossFunction):

        def forward(self, y_pred, y_true):
            return np.mean((y_true - y_pred) ** 2)