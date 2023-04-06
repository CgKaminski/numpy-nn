#
# Program1 - A Neural Network
# Due 02/14/2023
# By Cameron Kaminski
#

import resources as r
import numpy as np

hyper_params = r.parseHyperparams()

training_features = r.readDataToMatrix(hyper_params.train_feat, targets=False, type = hyper_params.type)
training_targets = r.readDataToMatrix(hyper_params.train_target, type = hyper_params.type)
dev_features = r.readDataToMatrix(hyper_params.dev_feat, targets=False, type = hyper_params.type)
dev_targets = r.readDataToMatrix(hyper_params.dev_target, type = hyper_params.type)
num_points = training_features.shape[0]
num_features = training_features.shape[1]

# Instantiate the Neural Network.
nn = r.NeuralNetwork(((np.zeros(num_features))), hyper_params.nunits, hyper_params.nlayers, hyper_params.init_range, hyper_params.num_classes, hyper_params.type, num_points, hyper_params.mb)

# Instantiate the activation functions.
nn.activation = r.activationFunction(hyper_params.hidden_act)
nn.output_activation = r.outputActivationFunction(hyper_params.num_classes, hyper_params.type)

batches = r.minibatch(training_features, training_targets, hyper_params.mb)

x = batches[0]
y = batches[1]

for epoch in range(hyper_params.epochs):

    batches = r.minibatch(training_features, training_targets, hyper_params.mb)

    X = batches[0]
    Y = batches[1]

    for i in range(len(X[0])):

        x = X[:,i,:]
        y = Y[:,i,:]

        # Gradient Descent
        nn.forward(x)
        nn.backwards(y, hyper_params.num_classes, hyper_params.learnrate)

        # Print the accuracy of the network.
        if hyper_params.v:
            train_accuracy = nn.checkAccuracy(training_features,
                                              training_targets, hyper_params.num_classes, hyper_params.
                                              type)
            dev_accuracy = nn.checkAccuracy(dev_features,
                                            dev_targets, hyper_params.num_classes, hyper_params.type)

            print(f"Update {str(i + 1).zfill(6)}: ", end=' ')
            print(f"train={round(train_accuracy, 3)} ", end=' ')
            print(f"dev={round(dev_accuracy, 3)} ")

    # Print the accuracy of the network.
    train_accuracy = nn.checkAccuracy(training_features,
                                      training_targets, hyper_params.num_classes, hyper_params.
                                      type)
    dev_accuracy = nn.checkAccuracy(dev_features,
                                    dev_targets, hyper_params.num_classes, hyper_params.type)
    print(f"Epoch {str(epoch + 1).zfill(6)}:", end=' ')
    print(f"train={round(train_accuracy, 3)}", end=' ')
    print(f"dev={round(dev_accuracy, 3)}")
