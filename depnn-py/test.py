import tensorflow as tf
import neuralnetwork

# Input

deps_dir = ""
model_dir = ""

model_path = model_dir + "/model"

# Code

network = Network(model_dir, train=False)
network.test(deps_dir)
