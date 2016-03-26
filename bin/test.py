#!/usr/bin/python

import tensorflow as tf
from depnn_py.neuralnetwork import Network

# Input

deps_dir = ""
model_dir = ""

model_path = model_dir + "/model"

# Code

network = Network(model_dir, train=False)
network.test(deps_dir)
