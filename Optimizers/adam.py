import numpy as np
from Base import BaseOptimizer

class Adam(BaseOptimizer):
    """
    Adam optimizer.

    >>> adam = Adam(learning_rate=0.001)
    >>> weights = np.array([[0.5, -0.5], [0.5, -0.5]])
    >>> gradients = np.array([[0.1, -0.2], [0.1, -0.2]])
    >>> for _ in range(1000):
    ...     adam.update(weights, gradients)
    >>> np.allclose(weights, np.array([[-0.5, 0.5], [-0.5, 0.5]]), atol=1e-2)
    True
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        """
        Update weights using the Adam optimization algorithm.

        :param weights: Current weights
        :param gradients: Computed gradients
        """
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Debug print statements to track shapes and values
        print(f"Iteration {self.t}")
        print(f"weights shape: {weights.shape}, weights: {weights}")
        print(f"gradients shape: {gradients.shape}, gradients: {gradients}")
        print(f"m shape: {self.m.shape}, m: {self.m}")
        print(f"v shape: {self.v.shape}, v: {self.v}")
        print(f"m_hat shape: {m_hat.shape}, m_hat: {m_hat}")
        print(f"v_hat shape: {v_hat.shape}, v_hat: {v_hat}")

        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
