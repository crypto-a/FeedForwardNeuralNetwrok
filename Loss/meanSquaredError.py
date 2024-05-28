from Base import BaseLossFunction
import numpy as np


class MeanSquaredError(BaseLossFunction):
    """
    Mean Squared Error loss function.

    >>> mse = MeanSquaredError()
    >>> y_pred = np.array([1.5, 2.0, 3.5])
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> np.isclose(mse.forward(y_pred, y_true), 0.16666667, atol=1e-7)
    True

    >>> y_pred = np.array([[0.5, 1.0], [1.5, 2.0]])
    >>> y_true = np.array([[0.0, 1.0], [1.0, 2.0]])
    >>> np.isclose(mse.forward(y_pred, y_true), 0.125, atol=1e-7)
    True

    >>> y_pred = np.array([1.5, 2.0, 3.5])
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> np.allclose(mse.backward(y_pred, y_true), np.array([0.33333333, 0.0, 0.33333333]), atol=1e-7)
    True

    >>> y_pred = np.array([[0.5, 1.0], [1.5, 2.0]])
    >>> y_true = np.array([[0.0, 1.0], [1.0, 2.0]])
    >>> np.allclose(mse.backward(y_pred, y_true), np.array([[0.25, 0.0], [0.25, 0.0]]), atol=1e-7)
    True
    """

    def forward(self, y_pred, y_true):
        """
        Calculate the sample losses given model output and true labels.

        :param y_pred: Predicted values
        :param y_true: True values
        :return: Mean squared error

        >>> mse = MeanSquaredError()
        >>> y_pred = np.array([1.5, 2.0, 3.5])
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> np.isclose(mse.forward(y_pred, y_true), 0.16666667, atol=1e-7)
        True

        >>> y_pred = np.array([[0.5, 1.0], [1.5, 2.0]])
        >>> y_true = np.array([[0.0, 1.0], [1.0, 2.0]])
        >>> np.isclose(mse.forward(y_pred, y_true), 0.125, atol=1e-7)
        True
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        """
        Calculate the gradient of the loss function with respect to the model output.

        :param y_pred: Predicted values
        :param y_true: True values
        :return: Gradient of the loss function

        >>> mse = MeanSquaredError()
        >>> y_pred = np.array([1.5, 2.0, 3.5])
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> np.allclose(mse.backward(y_pred, y_true), np.array([0.33333333, 0.0, 0.33333333]), atol=1e-7)
        True

        >>> y_pred = np.array([[0.5, 1.0], [1.5, 2.0]])
        >>> y_true = np.array([[0.0, 1.0], [1.0, 2.0]])
        >>> np.allclose(mse.backward(y_pred, y_true), np.array([[0.25, 0.0], [0.25, 0.0]]), atol=1e-7)
        True
        """
        return 2 * (y_pred - y_true) / y_true.size


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)