#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
from depnn_py.dependency import Dependency
from depnn_py.transdependency import TransDependency
import sys
import logging

# Input

test_dir = sys.argv[1]
model_dir = sys.argv[2]
nn_type = sys.argv[3]
log_file = sys.argv[4]

model_path = model_dir + "/model"

# Code

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(message)s")

try:
    logging.info("Initializing network")

    if nn_type == "dep":
        network = Network(model_dir, False, Dependency())
    elif nn_type == "transdep":
        network = Network(model_dir, False, TransDependency())
    else:
        raise ValueError("Invalid nnType")

    logging.info("Network initialized")
    network.test(test_dir, log_file)
except Exception as e:
    logging.exception("Exception on main handler")
