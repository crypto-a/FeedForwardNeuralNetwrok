from Base import BaseLossFunction
import numpy as np


class CategoricalCrossEntropy(BaseLossFunction):
    """
    Categorical cross-entropy loss.

    >>> cce = CategoricalCrossEntropy()
    >>> y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
    >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
    >>> np.allclose(cce.forward(y_pred, y_true), np.array([0.51082562, 0.69314718]), atol=1e-7)
    True

    >>> y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
    >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
    >>> np.allclose(cce.backward(y_pred, y_true), np.array([[0.05, -0.2, 0.15], [0.15, 0.1, -0.25]]), atol=1e-7)
    True
    """

    def forward(self, y_pred, y_true):
        """
        Calculate the sample losses given model output and true labels.

        :param y_pred: Predicted probabilities (softmax output) of shape (n_samples, n_classes)
        :param y_true: True labels, one-hot encoded of shape (n_samples, n_classes)
        :return: Sample losses for each instance in the batch

        >>> cce = CategoricalCrossEntropy()
        >>> y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
        >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
        >>> np.allclose(cce.forward(y_pred, y_true), np.array([0.51082562, 0.69314718]), atol=1e-7)
        True
        """
        sample_losses = -np.log(y_pred[range(y_true.shape[0]), y_true.argmax(axis=1)])
        return sample_losses

    def backward(self, y_pred, y_true):
        """
        Calculate the gradient of the loss function with respect to the model output.

        :param y_pred: Predicted probabilities (softmax output) of shape (n_samples, n_classes)
        :param y_true: True labels, one-hot encoded of shape (n_samples, n_classes)
        :return: Gradient of the loss function with respect to the predicted probabilities

        >>> cce = CategoricalCrossEntropy()
        >>> y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
        >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
        >>> np.allclose(cce.backward(y_pred, y_true), np.array([[0.05, -0.2, 0.15], [0.15, 0.1, -0.25]]), atol=1e-7)
        True
        """
        samples = y_true.shape[0]
        y_pred[range(samples), y_true.argmax(axis=1)] -= 1
        return y_pred / samples


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

