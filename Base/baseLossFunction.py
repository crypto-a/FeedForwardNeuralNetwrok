import numpy as np


class BaseLossFunction:
    def calculate(self, y_pred, y_true):
        """
        Calculate the data loss given model output and true labels

        :param y_pred:
        :param y_true:
        :return:
        """
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true):
        """
        Calculate the sample losses given model output and true labels
        :param y_pred:
        :param y_true:
        :return:
        """
        raise NotImplementedError

    def backward(self, d_values, y_true):
        """
        Calculate the gradient of the loss function with respect to the model output
        :param d_values:
        :param y_true:
        :return:
        """
        raise NotImplementedError