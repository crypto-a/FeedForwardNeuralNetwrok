


class BaseOptimizer:
    """
    Base class for all optimizers in the package.
    """
    learning_rate: float

    def update_parameters(self, layer):
        """
        Update the parameters of the layer.
        """
        raise NotImplementedError