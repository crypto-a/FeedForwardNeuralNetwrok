from .base.base import BaseActivationFunction


class Relu(BaseActivationFunction):
    """
    ReLU activation function.
    """

    def forward(self, x):
        """
        Forward pass of the activation function.
        :param x:
        :return:
        """
        return x * (x > 0)

    def backward(self, y):
        """
        Backward pass of the activation function.
        :param x:
        :return:
        """
        return y * (y > 0)

    def __str__(self):
        """
        String representation of the activation function.
        :return:
        """
        return self.__class__.__name__