#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
import sys

# Input

test_dir = sys.argv[1]
model_dir = sys.argv[2]

model_path = model_dir + "/model"

# Code

print "Initializing network"
network = Network(model_dir, train=False)
print "Network initialized"
network.test(test_dir)
