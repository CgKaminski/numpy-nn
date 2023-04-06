#
# System Arguments Function
# By: Cameron Kaminski
#
# This file contains the function that takes the system arguments and returns a
# dictionary of the arguments.
#

import argparse


def parseHyperparams():
    """
    Takes the system arguments and returns a dictionary of the arguments.

    @return args : dict
        A dictionary of the system arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true", help="Verbosity flag")
    parser.add_argument("-train_feat", type=str, required=True, help="Path to training feature file")
    parser.add_argument("-train_target", type=str, required=True, help="Path to training target file")
    parser.add_argument("-dev_feat", type=str, required=True, help="Path to development feature file")
    parser.add_argument("-dev_target", type=str, required=True, help="Path to development target file")
    parser.add_argument("-epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("-learnrate", type=float, required=True, help="Learning rate")
    parser.add_argument("-nunits", type=int, required=True, help="Number of hidden units")
    parser.add_argument("-type", type=str, required=True, help="Problem mode (binary/multiclass)")
    parser.add_argument("-hidden_act", type=str, required=True, help="Hidden unit activation function")
    parser.add_argument("-init_range", type=float, required=True, help="Initialization range")
    parser.add_argument("-num_classes", type=int, default=2, help="Number of classes (default: 2)")
    parser.add_argument("-mb", type=int, default=32, help="Minibatch size (default: 32)")
    parser.add_argument("-nlayers", type=int, default=1, help="Number of hidden layers (default: 1)")
    return parser.parse_args()
