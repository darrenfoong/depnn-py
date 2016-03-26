#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network

# Input

deps_dir = sys.argv[1]
model_dir = sys.argv[2]
prev_model = sys.argv[3]

# Code

network = Network(prev_model, train=True)
network.train(deps_dir, model_dir)
