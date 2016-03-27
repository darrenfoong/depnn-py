# adapted from https://github.com/aymericdamien/TensorFlow-Examples/

import numpy as np
import tensorflow as tf
import sklearn.metrics
import os
import logging
from wordvectors import WordVectors
from embeddings import Embeddings
from dataset import Dataset

# Parameters

w2v_layer_size = 50

nn_epochs = 1
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

num_cores = 20

# Code

class Network:
    def __init__(self, path, train):
        self.train_bool = train

        if self.train_bool:
            self.prev_model = path
            logging.info("Using previous word2vec model: " + self.prev_model)
            self.word_vectors = WordVectors(self.prev_model, w2v_layer_size, "UNKNOWN")
        else:
            self.model_dir = path
            self.word_vectors = WordVectors(self.model_dir + "/word2vec.txt", w2v_layer_size, "UNKNOWN")

        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])
        self.input_keep_prob = tf.placeholder("float")
        self.hidden_keep_prob = tf.placeholder("float")

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
        input_layer_drop = tf.nn.dropout(_X, self.input_keep_prob)
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer_drop, _weights["h"]), _biases["b"]))
        hidden_layer_drop = tf.nn.dropout(hidden_layer, self.hidden_keep_prob)
        return tf.matmul(hidden_layer, _weights["out"]) + _biases["out"]

    def train(self, deps_dir, model_dir):
        logging.info("Training network using " + deps_dir)

        dataset = Dataset(self, deps_dir, nn_batch_size)

        self.cat_embeddings = Embeddings(dataset.cat_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self.slot_embeddings = Embeddings(dataset.slot_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self.dist_embeddings = Embeddings(dataset.dist_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self.pos_embeddings = Embeddings(dataset.pos_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.y))
        optimizer = tf.train.AdagradOptimizer(learning_rate=nn_learning_rate).minimize(cost)
        grads_wrt_input_op = tf.gradients(cost, self.x)[0]

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=num_cores, intra_op_parallelism_threads=num_cores)) as sess:
            sess.run(init)

            for epoch in range(1, nn_epochs+1):
                curr_batch = 1

                while True:
                    next_batch = dataset.next()
                    if not next_batch:
                        break

                    batch_xs, batch_ys, deps_in_batch = next_batch

                    logging.info("Training batch " + str(epoch) + "/" + str(curr_batch))

                    _, grads_wrt_input = sess.run([optimizer, grads_wrt_input_op], feed_dict={self.x: batch_xs, self.y: batch_ys, self.input_keep_prob: nn_dropout, self.hidden_keep_prob: nn_dropout})

                    logging.info("Network updated")

                    for i in range(len(deps_in_batch)):
                        dep = deps_in_batch[i]
                        grad_wrt_input = nn_learning_rate * grads_wrt_input[i]
                        self.cat_embeddings.update(dep[1], grad_wrt_input, 1 * w2v_layer_size)
                        self.slot_embeddings.update(dep[2], grad_wrt_input, 2 * w2v_layer_size)
                        self.dist_embeddings.update(dep[4], grad_wrt_input, 4 * w2v_layer_size)
                        self.pos_embeddings.update(dep[5], grad_wrt_input, 5 * w2v_layer_size)
                        self.pos_embeddings.update(dep[6], grad_wrt_input, 6 * w2v_layer_size)
                        self.pos_embeddings.update(dep[7], grad_wrt_input, 7 * w2v_layer_size)
                        self.pos_embeddings.update(dep[8], grad_wrt_input, 8 * w2v_layer_size)
                        self.pos_embeddings.update(dep[9], grad_wrt_input, 9 * w2v_layer_size)
                        self.pos_embeddings.update(dep[10], grad_wrt_input, 10 * w2v_layer_size)

                    logging.info("Embeddings updated")

                    curr_cost = sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys, self.input_keep_prob: nn_dropout, self.hidden_keep_prob: nn_dropout})

                    logging.info("Cost: " + str(curr_cost))

                    curr_batch += 1

                logging.info("Serializing network")
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

            logging.info("Network training complete")

    def test(self, deps_dir):
        dataset = Dataset(self, deps_dir, 0)

        self.cat_embeddings = Embeddings(self.model_dir + "/cat.emb", False, w2v_layer_size)
        self.slot_embeddings = Embeddings(self.model_dir + "/slot.emb", False, w2v_layer_size)
        self.dist_embeddings = Embeddings(self.model_dir + "/dist.emb", False, w2v_layer_size)
        self.pos_embeddings = Embeddings(self.model_dir + "/pos.emb", False, w2v_layer_size)

        saver = tf.train.Saver()

        model_path = self.model_dir + "/model.out"

        with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=num_cores, intra_op_parallelism_threads=num_cores)) as sess:
            saver.restore(sess, model_path)

            batch_xs, batch_ys, _ = dataset.next()

            correct_prediction = tf.equal(tf.argmax(self.network,1), tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            y_p = tf.argmax(self.network, 1)
            val_accuracy, y_network = sess.run([accuracy, y_p], feed_dict={self.x: batch_xs, self.y: batch_ys, self.input_keep_prob: 1.0, self.hidden_keep_prob: 1.0})

            y_true = np.argmax(batch_ys, 1)

            logging.info("Accuracy: " + str(val_accuracy))

            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_network)

            logging.info("Examples labeled as 0 classified by model as 0: " + str(confusion_matrix[0][0]))
            logging.info("Examples labeled as 0 classified by model as 1: " + str(confusion_matrix[0][1]))
            logging.info("Examples labeled as 1 classified by model as 0: " + str(confusion_matrix[1][0]))
            logging.info("Examples labeled as 1 classified by model as 1: " + str(confusion_matrix[1][1]))

        logging.info("Network testing complete")
