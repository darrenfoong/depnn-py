import tensorflow as tf
import neuralnetwork

# Input

deps_dir = ""
prev_model = ""
model_dir = ""

# Code

network = Network(prev_model, train=True)
network.train(deps_dir, model_dir)
