from Base import BaseLossFunction
import numpy as np

class CategoricalCrossEntropy(BaseLossFunction):
    """
    Categorical cross-entropy loss.
    """

    def forward(self, y_pred, y_true):
        """
        Calculate the sample losses given model output and true labels.
        :param y_pred: Predicted probabilities (softmax output) of shape (n_samples, n_classes)
        :param y_true: True labels, one-hot encoded of shape (n_samples, n_classes)
        :return: Sample losses for each instance in the batch
        """
        epsilon = 1e-10  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        sample_losses = -np.log(y_pred[range(y_true.shape[0]), y_true.argmax(axis=1)])
        return sample_losses

    def backward(self, y_pred, y_true):
        """
        Calculate the gradient of the loss function with respect to the model output.
        :param y_pred: Predicted probabilities (softmax output) of shape (n_samples, n_classes)
        :param y_true: True labels, one-hot encoded of shape (n_samples, n_classes)
        :return: Gradient of the loss function with respect to the predicted probabilities
        """
        samples = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1. - 1e-10)  # Clip predictions to avoid division by zero
        y_pred[range(samples), y_true.argmax(axis=1)] -= 1
        return y_pred / samples

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
