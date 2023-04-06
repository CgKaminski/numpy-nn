#
# Mini-batching
# By: Cameron Kaminski
#
# This file contains the code for the mini-batching the data.
#

import numpy as np


def minibatch(features, targets, mb) -> tuple:
    """
    Mini-batching the data.
    @param features : numpy.ndarray
    @param targets : numpy.ndarray
    @param mb : int
    @return : numpy.ndarray
    """
    n_examples = features.shape[0]
    indices = np.arange(n_examples)
    np.random.shuffle(indices)

    if mb == 0:
        features = features[indices].reshape(1, -1, features.shape[1])
        targets = targets[indices].reshape(1, -1, targets.shape[1])
        return features, targets

    feature_mb = np.zeros((n_examples // mb, mb, features.shape[1]))
    target_mb = np.zeros((n_examples // mb, mb, targets.shape[1]))

    for i in range(n_examples // mb):
        feature_mb[i] = features[indices[i * mb: (i + 1) * mb]]
        target_mb[i] = targets[indices[i * mb: (i + 1) * mb]]

    return feature_mb, target_mb
