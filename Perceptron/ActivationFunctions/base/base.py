
class BaseActivationFunction:
    """
    Base class for activation functions in neural networks.
    """

    def forward(self, x):
        """
        Forward pass of the activation function.
        :param x:
        :return:
        """
        raise NotImplementedError

    def backward(self, x):
        """
        Backward pass of the activation function.
        :param x:
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        """
        String representation of the activation function.
        :return:
        """
        return self.__class__.__name__