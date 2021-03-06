#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
from depnn_py.dependency import Dependency
from depnn_py.longdependency import LongDependency
from depnn_py.transdependency import TransDependency
from depnn_py.feature import Feature
import sys
import logging

# Input

train_dir = sys.argv[1]
model_dir = sys.argv[2]
prev_model = sys.argv[3]
nn_type = sys.argv[4]
log_file = sys.argv[5]

# Code

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(message)s")

try:
    logging.info("Initializing network")

    if nn_type == "dep":
        network = Network(prev_model, True, Dependency())
    elif nn_type == "longdep":
        network = Network(prev_model, True, LongDependency())
    elif nn_type == "transdep":
        network = Network(prev_model, True, TransDependency())
    elif nn_type == "feature":
        network = Network(prev_model, True, Feature())
    else:
        raise ValueError("Invalid nnType")

    logging.info("Network initialized")
    network.train(train_dir, model_dir)
except Exception as e:
    logging.exception("Exception on main handler")
