#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network

# Input

test_dir = sys.argv[1]
model_dir = sys.argv[2]

model_path = model_dir + "/model"

# Code

network = Network(model_dir, train=False)
network.test(test_dir)
