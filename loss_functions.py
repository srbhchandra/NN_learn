"""
Module with loss functions and their derivatives with respect to h = np.dot(w.x) + b
The input to all functions y_score is the output of some activation function.
"""
import numpy as np

def binary_cross_entropy_loss(y: np.ndarray, y_score: np.ndarray) -> float:
    """
    Used typically with sigmoid activation.
    y is (n, 1) np.ndarray where n is the number of samples/datapoints.
    y_score is (n, 1) np.ndarray.
    """
    loss_i = y * np.log(y_score) + (1 - y) * np.log(1 - y_score)
    return -np.mean(loss_i)


def d_binary_cross_entropy_loss(y: np.ndarray, y_score: np.ndarray) -> float:
    """
    Used typically with sigmoid activation.
    y is (n, 1) np.ndarray where n is the number of samples/datapoints.
    y_score is (n, 1) np.ndarray.
    """
    return y_score - y


def multi_cross_entropy_loss(y: np.ndarray, y_score: np.ndarray) -> float:
    """
    Used typically with softmax activation.
    y is (n, 1) np.ndarray where n is the number of samples/datapoints.
    y_score is (n, c) np.ndarray where c is the number of classes.
    """
    loss_i = -np.log(exp[:, y])
    return np.mean(loss_i)


def d_multi_cross_entropy_loss(y: np.ndarray, y_score: np.ndarray) -> float:
    """
    Used typically with softmax activation.
    y is (n, 1) np.ndarray where n is the number of samples/datapoints.
    y_score is (n, c) np.ndarray where c is the number of classes.
    """
    loss_derivative = np.copy(y_score)
    loss_derivative[:, y] -= 1
    return loss_derivative
