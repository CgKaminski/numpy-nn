#
# Neural Network Class
# By: Cameron Kaminski
#
# This file contains the class implementation and the initializer functions for
# the Neural Network class.
# Note: the neural network class only contains the layers for the network.
# Note: the layers are encapsulated in 3 sections
#   - Input Layer
#   - Hidden Layer(s)
#   - Output layer
#


import numpy as np


class NeuralNetwork:

    def __init__(self, x: np.ndarray, nunits: int, nlayers: int,
                 range: float, classes: int, type: chr, num_points : int, mb : int):

        self.type = type
        self.activation = None
        self.output_activation = None
        if mb != 0:
            self.num_batches = num_points // mb
            self.mb = mb
        else:
            self.num_batches = 1
            self.mb = 1
        self.y = None
        self.w = [None] * (nlayers + 1)
        self.b = [None] * (nlayers + 1)
        self.z = [None] * (nlayers + 2)

        # Initializing Weights and Biases
        self.w[0] = np.random.uniform(-range, range, (nunits, x.shape[0]))
        self.b[0] = np.random.uniform(-range, range, (nunits, 1))
        self.w[1:-1] = np.random.uniform(-range, range, (nlayers - 1, nunits,
                                                         nunits))
        self.b[1:-1] = np.random.uniform(-range, range, (nlayers - 1, nunits, 1))
        self.w[-1] = np.random.uniform(-range, range, (classes, nunits))
        self.b[-1] = np.random.uniform(-1, 1, (classes, 1))

        # Initializing Z
        self.z[0] = x
        #self.z[1:-1] = np.zeros((nlayers, self.num_batches, nunits, 1))
        #self.z[-1] = np.zeros((classes, self.num_batches, 1))

    def setY(self, y: np.ndarray, classes: int) -> None:
        """
        Sets the y value for the neural network.
        @param y:
        @param classes:
        @return: None
        """

        # Classification case
        if self.type == 'C':
            y = y.astype(int)
            one_hot = np.zeros((classes, self.num_batches))
            for i in range(y.shape[0]):
                one_hot[y[i, 0], i] = 1
            self.y = one_hot
        # Regression case
        elif self.type == 'R':
            self.y = y.reshape(-1, self.num_batches)

    def setX(self, x: np.ndarray) -> None:
        """
        Sets the x value for the neural network.
        @param x:
        @return: None
        """
        self.z[0] = x.reshape(-1, self.num_batches)

    def backwards(self, y: np.ndarray, classes: int, lr: float) -> None:
        """
        Backpropagation algorithm for the neural network.
        @param y:
        @param classes:
        @param lr:
        @return: None
        """

        self.setY(y, classes)

        dw = [None] * len(self.w)
        db = [None] * len(self.b)
        deltas = [None] * len(self.w)

        # Calculate delta errors
        deltas[-1] = self.output_activation(self.z[-1]) - self.y
        for i in range(len(self.w) - 2, -1, -1):
            deltas[i] = self.activation(self.z[i + 1], True) * self.w[i + 1].T.dot(deltas[i + 1])

        # Calculate weight and bias changes
        for i in range(1, len(deltas)):
            dw[i] = deltas[i] @ self.activation(self.z[i]).T
            db[i] = deltas[i] @ np.ones((self.num_batches, 1))
        dw[0] = deltas[0] @ self.z[0].T
        db[0] = deltas[0] @ np.ones((self.num_batches, 1))

        # Update weights and biases
        for i in range(len(self.w)):
            self.w[i] -= lr * dw[i] / self.mb
            self.b[i] -= lr * db[i] / self.mb

    def forward(self, x: np.ndarray) -> None:
        """
        Forward propagation algorithm for the neural network.
        @param x:
        @return: None
        """

        self.setX(x)

        # Forward feed
        self.z[1] = self.w[0].dot(self.z[0]) + self.b[0]
        for i in range(2, len(self.z), 1):
            self.z[i] = self.w[i - 1].dot(self.activation(self.z[i - 1])) + self.b[i - 1]

    def checkAccuracy(self, X: np.ndarray, Y: np.ndarray, classes: int,
                      type: chr) -> float:
        """
        Checks the accuracy of the neural network.
        @param X:
        @param Y:
        @param classes:
        @param type:
        @return: float
        """

        # Storing old number of batches
        old_num_batches = self.num_batches
        old_mb = self.mb
        self.num_batches = 1
        self.mb = 1
        # Case for classification

        if type == 'C':  # Accuracy
            Y = Y.reshape(-1, )
            predictions = np.zeros((len(Y)))
            for i in range(len(Y)):
                x = X[i]
                self.forward(x)
                a = self.output_activation(self.z[-1])
                prediction = np.argmax(a)
                predictions[i] = prediction

            self.num_batches = old_num_batches
            self.mb = old_mb
            return np.sum(predictions == Y) / len(Y)

        # Case for a single linear regression
        elif type == 'R' and classes == 1:  # MSE Loss
            predictions = np.zeros((len(Y)))
            for i in range(len(Y)):
                x = X[i]
                self.forward(x)
                a = self.output_activation(self.z[-1])
                predictions[i] = a
            self.num_batches = old_num_batches
            self.mb = old_mb
            return np.sum((Y - predictions) ** 2) / len(Y)

        # Case for multi linear regression
        elif type == 'R' and classes > 1:  # MSE Loss
            predictions = np.zeros((len(Y), classes))
            for i in range(len(Y)):
                x = X[i]
                self.forward(x)
                a = self.output_activation(self.z[-1])
                predictions[i] = a.reshape(-1, )
            self.num_batches = old_num_batches
            self.mb = old_mb
            return np.sum((Y - predictions) ** 2) / len(Y)