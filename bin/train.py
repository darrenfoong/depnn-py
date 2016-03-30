#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
import sys
import logging

# Input

deps_dir = sys.argv[1]
model_dir = sys.argv[2]
prev_model = sys.argv[3]
log_file = sys.argv[4]

# Code

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(message)s")

try:
    logging.info("Initializing network")
    network = Network(prev_model, train=True)
    logging.info("Network initialized")
    network.train(deps_dir, model_dir)
except Exception as e:
    logging.exception("Exception on main handler")
