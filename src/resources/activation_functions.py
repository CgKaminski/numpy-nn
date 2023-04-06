#
# Activation Functions
# By: Cameron Kaminski
# 
#
# This file contains the activation functions that used for the neural network.
# The activation functions are:
#   - Sigmoid
#   - Tanh
#   - Softmax
#   - ReLU
#   - Identity
# Additionally, this file contains the activationFunction function, which takes
# a layer and an activation function and applies the activation function to the
# layer.
#

import numpy as np


def outputActivationFunction(num_classes: int, type: chr):
    """
    Takes the number of classes and return the activation function for the
    output layer.
    @param type:
    @param num_classes : int
    @return activation_function : function
    """

    if num_classes == 2:
        activation_function = "sig"
    else:
        activation_function = "softmax"
    if type == "R":
        activation_function = "identity"
    return activationFunction(activation_function)


def activationFunction(activation: str):
    """
    Takes an activation function name and returns the activation function.
    @param activation: str
    @return function : function
    """
    if activation == "sig":
        activated_function = sigmoidActivation
    elif activation == "tanh":
        activated_function = tanhActivation
    elif activation == "softmax":
        activated_function = softmaxActivation
    elif activation == "relu":
        activated_function = reluActivation
    elif activation == "identity":
        activated_function = identityActivation
    else:
        raise TypeError("Invalid activation function: " +
                        str(activation))

    return activated_function


def sigmoidActivation(layer: np.ndarray, derivative=False) -> float:
    """
    Takes an array, and applies the sigmoidal activation function.
    @param layer : numpy.ndarray
    @param derivative : bool
    @return numpy.ndarray
    """
    if derivative:
        return sigmoidActivation(layer) * (1 - sigmoidActivation(layer))
    else:
        return 1 / (1 + np.exp(-1 * layer))


def tanhActivation(layer: np.ndarray, derivative=False) -> float:
    """
    Takes an array, and applies the tanh activation function.
    @param layer : numpy.ndarray
    @param derivative : bool
    @return numpy.ndarray
    """
    if derivative:
        return 1 - tanhActivation(layer) ** 2
    else:
        return 2 / (1 + np.exp(-2 * layer)) - 1


def softmaxActivation(layer: np.ndarray, derivative=False) -> float:
    """
    Takes an array and applies the softmax function.
    @param layer : numpy.ndarray
    @param derivative : bool
    @return numpy.ndarray
    """
    if derivative:
        return softmaxActivation(layer) * (1 - softmaxActivation(layer))
    else:
        layer = layer.T
        layer = layer - np.max(layer, axis=1, keepdims=True)
        exp_layer = np.exp(layer)
        sum_exp = np.sum(exp_layer, axis=1, keepdims=True)
        return exp_layer.T / sum_exp.T


def reluActivation(layer: np.ndarray, derivative=False):
    """
    Takes a layer and applies the relu activation function.
    @param layer : numpy.ndarray
    @param derivative : bool
    @return numpy.ndarray
    """
    if derivative:
        return np.where(layer > 0, 1, 0)
    else:
        return np.maximum(0, layer)


def identityActivation(layer, derivative=False):
    """
    Takes a layer and applies the identity activation function.
    @param derivative: bool
    @param layer : numpy.ndarray
    @return np.ndarray
    """

    return layer
