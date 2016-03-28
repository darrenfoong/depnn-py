# adapted from https://github.com/aymericdamien/TensorFlow-Examples/

import numpy as np
import tensorflow as tf
import sklearn.metrics
import math
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

# Code

class Network:
    def __init__(self, path, train):
        self._train_bool = train

        if self._train_bool:
            self._prev_model = path
            logging.info("Using previous word2vec model: " + self._prev_model)
            self._word_vectors = WordVectors(self._prev_model, w2v_layer_size, "UNKNOWN")
        else:
            self._model_dir = path
            self._word_vectors = WordVectors(self._model_dir + "/word2vec.txt", w2v_layer_size, "UNKNOWN")

        self._x = tf.placeholder("float", [None, n_input])
        self._y = tf.placeholder("float", [None, n_classes])
        self._input_keep_prob = tf.placeholder("float")
        self._hidden_keep_prob = tf.placeholder("float")

        # ReLU
        w_h_stddev = math.sqrt(2 / n_input)

        # Xavier
        w_out_stddev = math.sqrt(3 / (nn_hidden_layer_size + n_classes))

        self._weights = {
            "h": tf.Variable(tf.truncated_normal([n_input, nn_hidden_layer_size], stddev=w_h_stddev), name="w_h"),
            "out": tf.Variable(tf.truncated_normal([nn_hidden_layer_size, n_classes], stddev=w_out_stddev), name="w_out")
        }

        self._biases = {
            "b": tf.Variable(tf.constant(0.1, shape=[nn_hidden_layer_size]), name="b_b"),
            "out": tf.Variable(tf.random_normal([n_classes]), name="b_out")
        }

        self._network = self._multilayer_perceptron(self._x, self._weights, self._biases)

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

        head_vector = self._word_vectors.get(head.lower())
        dependent_vector = self._word_vectors.get(dependent.lower())

        category_vector = self._cat_embeddings.get(category)
        slot_vector = self._slot_embeddings.get(slot)
        distance_vector = self._dist_embeddings.get(distance)
        head_pos_vector = self._pos_embeddings.get(head_pos)
        dependent_pos_vector = self._pos_embeddings.get(dependent_pos)
        head_left_pos_vector = self._pos_embeddings.get(head_left_pos)
        head_right_pos_vector = self._pos_embeddings.get(head_right_pos)
        dependent_left_pos_vector = self._pos_embeddings.get(dependent_left_pos)
        dependent_right_pos_vector = self._pos_embeddings.get(dependent_right_pos)

        return np.hstack((head_vector, category_vector, slot_vector, dependent_vector, distance_vector, head_pos_vector, dependent_pos_vector, head_left_pos_vector, head_right_pos_vector, dependent_left_pos_vector, dependent_right_pos_vector))

    def _multilayer_perceptron(self, _X, _weights, _biases):
        input_layer_drop = tf.nn.dropout(_X, self._input_keep_prob)
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer_drop, _weights["h"]), _biases["b"]))
        hidden_layer_drop = tf.nn.dropout(hidden_layer, self._hidden_keep_prob)
        return tf.matmul(hidden_layer, _weights["out"]) + _biases["out"]

    def train(self, deps_dir, model_dir):
        logging.info("Training network using " + deps_dir)

        dataset = Dataset(self, deps_dir, nn_batch_size)

        self._cat_embeddings = Embeddings(dataset.cat_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self._slot_embeddings = Embeddings(dataset.slot_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self._dist_embeddings = Embeddings(dataset.dist_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)
        self._pos_embeddings = Embeddings(dataset.pos_lexicon, True, w2v_layer_size, random_range=nn_embed_random_range)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._network, self._y))
        regularizers = tf.nn.l2_loss(self._weights["h"]) + tf.nn.l2_loss(self._weights["out"]) + tf.nn.l2_loss(self._biases["b"]) + tf.nn.l2_loss(self._biases["out"])
        cost += nn_l2_reg * regularizers

        optimizer = tf.train.AdagradOptimizer(learning_rate=nn_learning_rate).minimize(cost)
        grads_wrt_input_op = tf.gradients(cost, self._x)[0]

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(1, nn_epochs+1):
                logging.info("Training epoch " + str(epoch))

                curr_batch = 1
                sum_cost = 0

                while True:
                    next_batch = dataset.next()
                    if not next_batch:
                        break

                    batch_xs, batch_ys, deps_in_batch = next_batch

                    logging.info("Training batch " + str(epoch) + "/" + str(curr_batch))

                    _, grads_wrt_input = sess.run([optimizer, grads_wrt_input_op], feed_dict={self._x: batch_xs, self._y: batch_ys, self._input_keep_prob: nn_dropout, self._hidden_keep_prob: nn_dropout})

                    logging.info("Network updated")

                    for i in range(len(deps_in_batch)):
                        dep = deps_in_batch[i]
                        grad_wrt_input = nn_learning_rate * grads_wrt_input[i]
                        self._cat_embeddings.update(dep[1], grad_wrt_input, 1 * w2v_layer_size)
                        self._slot_embeddings.update(dep[2], grad_wrt_input, 2 * w2v_layer_size)
                        self._dist_embeddings.update(dep[4], grad_wrt_input, 4 * w2v_layer_size)
                        self._pos_embeddings.update(dep[5], grad_wrt_input, 5 * w2v_layer_size)
                        self._pos_embeddings.update(dep[6], grad_wrt_input, 6 * w2v_layer_size)
                        self._pos_embeddings.update(dep[7], grad_wrt_input, 7 * w2v_layer_size)
                        self._pos_embeddings.update(dep[8], grad_wrt_input, 8 * w2v_layer_size)
                        self._pos_embeddings.update(dep[9], grad_wrt_input, 9 * w2v_layer_size)
                        self._pos_embeddings.update(dep[10], grad_wrt_input, 10 * w2v_layer_size)

                    logging.info("Embeddings updated")

                    curr_cost = sess.run(cost, feed_dict={self._x: batch_xs, self._y: batch_ys, self._input_keep_prob: nn_dropout, self._hidden_keep_prob: nn_dropout})

                    logging.info("Cost: " + str(curr_cost))

                    curr_batch += 1
                    sum_cost += curr_cost

                logging.info("Epoch cost: " + str(sum_cost/float(curr_batch-1)))

                model_epoch_dir = model_dir + "/epoch" + str(epoch)

                if not os.path.exists(model_epoch_dir):
                    os.makedirs(model_epoch_dir)

                self._serialize(saver, sess, model_epoch_dir)

                dataset.reset()

            self._serialize(saver, sess, model_dir)

            logging.info("Network training complete")

    def test(self, test_dir, log_file):
        logging.info("Testing network using " + test_dir)

        dataset = Dataset(self, test_dir, 0)

        self._cat_embeddings = Embeddings(self._model_dir + "/cat.emb", False, w2v_layer_size)
        self._slot_embeddings = Embeddings(self._model_dir + "/slot.emb", False, w2v_layer_size)
        self._dist_embeddings = Embeddings(self._model_dir + "/dist.emb", False, w2v_layer_size)
        self._pos_embeddings = Embeddings(self._model_dir + "/pos.emb", False, w2v_layer_size)

        saver = tf.train.Saver()

        model_path = self._model_dir + "/model.out"

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            batch_xs, batch_ys, deps_in_batch = dataset.next()

            logging.info("Number of test examples: " + str(len(deps_in_batch)))

            correct_prediction = tf.equal(tf.argmax(self._network,1), tf.argmax(self._y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            y_p = tf.argmax(self._network, 1)
            val_accuracy, y_network = sess.run([accuracy, y_p], feed_dict={self._x: batch_xs, self._y: batch_ys, self._input_keep_prob: 1.0, self._hidden_keep_prob: 1.0})

            y_true = np.argmax(batch_ys, 1)

            logging.info("Accuracy: " + str(val_accuracy))

            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_network)

            logging.info("Examples labeled as 0 classified by model as 0: " + str(confusion_matrix[0][0]))
            logging.info("Examples labeled as 0 classified by model as 1: " + str(confusion_matrix[0][1]))
            logging.info("Examples labeled as 1 classified by model as 0: " + str(confusion_matrix[1][0]))
            logging.info("Examples labeled as 1 classified by model as 1: " + str(confusion_matrix[1][1]))

        with open(log_file+".classified1", "w") as out_correct, \
             open(log_file+".classified0", "w") as out_incorrect:

            logging.info("Writing to files")

            for i in range(len(deps_in_batch)):
                prediction = y_network[i]

                if prediction >= 0.5:
                    out_correct.write(" ".join(deps_in_batch[i]) + "\n")
                else:
                    out_incorrect.write(" ".join(deps_in_batch[i]) + "\n")

        logging.info("Network testing complete")

    def _serialize(self, saver, sess, model_dir):
        logging.info("Serializing network")
        saver.save(sess, model_dir + "/model.out")
        logging.info("Serializing embeddings")
        self._cat_embeddings.serialize(model_dir + "/cat.emb")
        self._slot_embeddings.serialize(model_dir + "/slot.emb")
        self._dist_embeddings.serialize(model_dir + "/dist.emb")
        self._pos_embeddings.serialize(model_dir + "/pos.emb")
