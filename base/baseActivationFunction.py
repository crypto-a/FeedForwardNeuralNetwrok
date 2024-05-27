
class BaseActivationFunction:
    """
    This Object is the base class for all activation functions.
    It will have all the not implemented methods that will be implemented in the child classes.
    """

    input: float

    def forward(self, inputs):
        """
        This method will be implemented in the child classes.
        It will be used to calculate the forward pass of the activation function.
        :param x:
        :return:
        """
        raise NotImplementedError

    def backward(self, d_out):
        """
        This method will be implemented in the child classes.
        It will be used to calculate the backward pass of the activation function.
        :param y:
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __call__(self, x):
        return self.forward(x)