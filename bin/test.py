#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
import sys
import logging

# Input

test_dir = sys.argv[1]
model_dir = sys.argv[2]
log_file = sys.argv[3]

model_path = model_dir + "/model"

# Code

logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")

logging.info("Initializing network")
network = Network(model_dir, train=False)
logging.info("Network initialized")
network.test(test_dir)
