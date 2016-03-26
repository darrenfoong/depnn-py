#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network
import sys

# Input

deps_dir = sys.argv[1]
model_dir = sys.argv[2]
prev_model = sys.argv[3]

# Code

print "Initializing network"
network = Network(prev_model, train=True)
print "Network initialized"
network.train(deps_dir, model_dir)
