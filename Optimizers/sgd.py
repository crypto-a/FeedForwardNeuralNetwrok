from Base import BaseOptimizer

import numpy as np


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.

    >>> sgd = SGD(learning_rate=0.01)
    >>> weights = np.array([0.5, -0.5])
    >>> gradients = np.array([0.1, -0.2])
    >>> sgd.update(weights, gradients)
    >>> np.allclose(weights, np.array([0.499, -0.498]), atol=1e-7)
    True
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        """
        Update weights using the gradient descent algorithm.

        :param weights: Current weights
        :param gradients: Computed gradients
        """
        weights -= self.learning_rate * gradients


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
