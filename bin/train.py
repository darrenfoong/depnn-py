#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network

# Input

deps_dir = ""
prev_model = ""
model_dir = ""

# Code

network = Network(prev_model, train=True)
network.train(deps_dir, model_dir)
