# adapted from https://github.com/aymericdamien/TensorFlow-Examples/

import numpy as np
import tensorflow as tf
import sklearn.metrics
import os
from wordvectors import WordVectors
from embeddings import Embeddings
from dataset import Dataset

# Parameters

w2v_layer_size = 50

nn_epochs = 5
nn_batch_size = 128
nn_hidden_layer_size = 200
nn_learning_rate = 1e-2
nn_l2_reg = 1e-8
nn_dropout = 0.5
nn_embed_random_range = 0.01
nn_hard_labels = True

n_properties = 11
n_input = n_properties * w2v_layer_size
n_classes = 2

# Code

class Network:
    def __init__(self, path, train):
        self.train_bool = train

        if self.train_bool:
            self.prev_model = path
            print "Using previous word2vec model: " + self.prev_model
            self.word_vectors = WordVectors(self.prev_model, "UNKNOWN")
        else:
            self.model_dir = path
            self.word_vectors = WordVectors(self.model_dir + "/word2vec.txt", "UNKNOWN")

        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        self.weights = {
            "h": tf.Variable(tf.random_normal([n_input, nn_hidden_layer_size]), name="w_h"),
            "out": tf.Variable(tf.random_normal([nn_hidden_layer_size, n_classes]), name="w_out")
}
        self.biases = {
            "b": tf.Variable(tf.random_normal([nn_hidden_layer_size]), name="b_b"),
            "out": tf.Variable(tf.random_normal([n_classes]), name="b_out")
}
        self.network = self.multilayer_perceptron(self.x, self.weights, self.biases)

    def make_vector(self, dep):
        head = dep[0]
        category = dep[1]
        slot = dep[2]
        dependent = dep[3]
        distance = dep[4]
        head_pos = dep[5]
        dependent_pos = dep[6]
        head_left_pos = dep[7]
        head_right_pos = dep[8]
        dependent_left_pos = dep[9]
        dependent_right_pos = dep[10]

        head_vector = self.word_vectors.get(head.lower())
        dependent_vector = self.word_vectors.get(dependent.lower())

        category_vector = self.cat_embeddings.get(category)
        slot_vector = self.slot_embeddings.get(slot)
        distance_vector = self.dist_embeddings.get(distance)
        head_pos_vector = self.pos_embeddings.get(head_pos)
        dependent_pos_vector = self.pos_embeddings.get(dependent_pos)
        head_left_pos_vector = self.pos_embeddings.get(head_left_pos)
        head_right_pos_vector = self.pos_embeddings.get(head_right_pos)
        dependent_left_pos_vector = self.pos_embeddings.get(dependent_left_pos)
        dependent_right_pos_vector = self.pos_embeddings.get(dependent_right_pos)

        return np.hstack((head_vector, category_vector, slot_vector, dependent_vector, distance_vector, head_pos_vector, dependent_pos_vector, head_left_pos_vector, head_right_pos_vector, dependent_left_pos_vector, dependent_right_pos_vector))

    def multilayer_perceptron(self, _X, _weights, _biases):
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(_X, _weights["h"]), _biases["b"]))
        return tf.matmul(hidden_layer, _weights["out"]) + _biases["out"]

    def train(self, deps_dir, model_dir):
        print "Training network using " + deps_dir

        dataset = Dataset(self, deps_dir, nn_batch_size)

        self.cat_embeddings = Embeddings(dataset.cat_lexicon, train=True, random_range=nn_embed_random_range)
        self.slot_embeddings = Embeddings(dataset.slot_lexicon, train=True, random_range=nn_embed_random_range)
        self.dist_embeddings = Embeddings(dataset.dist_lexicon, train=True, random_range=nn_embed_random_range)
        self.pos_embeddings = Embeddings(dataset.pos_lexicon, train=True, random_range=nn_embed_random_range)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=nn_learning_rate).minimize(cost)

        # TODO AdaGrad

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(1, nn_epochs+1):
                avg_cost = 0
                curr_batch = 1
                num_batch = 0

                while True:
                    next_batch = dataset.next()
                    if not next_batch:
                        break

                    batch_xs, batch_ys = next_batch

                    print "Training batch " + str(epoch) + "/" + str(curr_batch)
                    sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})

                    # TODO how to track num_batch across epochs?
                    avg_cost += sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys})/curr_batch

                    print "Cost: " + str(avg_cost)

                    # TODO get errors and update embeddings

                    curr_batch += 1

                print "Serializing network"
                model_epoch_dir = model_dir + "/epoch" + str(epoch)

                if not os.path.exists(model_epoch_dir):
                    os.makedirs(model_epoch_dir)

                saver.save(sess, model_epoch_dir + "/model.out")
                self.cat_embeddings.serialize(model_epoch_dir + "/cat.emb")
                self.slot_embeddings.serialize(model_epoch_dir + "/slot.emb")
                self.dist_embeddings.serialize(model_epoch_dir + "/dist.emb")
                self.pos_embeddings.serialize(model_epoch_dir + "/pos.emb")

                dataset.reset()

            saver.save(sess, model_dir + "/model.out")
            self.cat_embeddings.serialize(model_dir + "/cat.emb")
            self.slot_embeddings.serialize(model_dir + "/slot.emb")
            self.dist_embeddings.serialize(model_dir + "/dist.emb")
            self.pos_embeddings.serialize(model_dir + "/pos.emb")

            print "Network training complete"

    def test(self, deps_dir):
        dataset = Dataset(self, deps_dir, 0)

        self.cat_embeddings = Embeddings(self.model_dir + "/cat.emb", train=False)
        self.slot_embeddings = Embeddings(self.model_dir + "/slot.emb", train=False)
        self.dist_embeddings = Embeddings(self.model_dir + "/dist.emb", train=False)
        self.pos_embeddings = Embeddings(self.model_dir + "/pos.emb", train=False)

        saver = tf.train.Saver()

        model_path = self.model_dir + "/model.out"

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            batch_xs, batch_ys = dataset.next()

            correct_prediction = tf.equal(tf.argmax(self.network,1), tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            y_p = tf.argmax(self.network, 1)
            val_accuracy, y_network = sess.run([accuracy, y_p], feed_dict={self.x: batch_xs, self.y: batch_ys})

            y_true = np.argmax(batch_ys, 1)

            print "Accuracy:", val_accuracy

            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_network)

            print "Examples labeled as 0 classified by model as 0: " + str(confusion_matrix[0][0])
            print "Examples labeled as 0 classified by model as 1: " + str(confusion_matrix[0][1])
            print "Examples labeled as 1 classified by model as 0: " + str(confusion_matrix[1][0])
            print "Examples labeled as 1 classified by model as 1: " + str(confusion_matrix[1][1])

        print "Network testing complete"
