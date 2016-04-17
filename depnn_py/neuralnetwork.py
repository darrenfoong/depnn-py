# adapted from https://github.com/aymericdamien/TensorFlow-Examples/

import numpy as np
import tensorflow as tf
import sklearn.metrics
import math
import os
import struct
import codecs
import sys
import logging
from wordvectors import WordVectors
from embeddings import Embeddings
from datasetiterator import DataSetIterator

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

n_properties = None
n_input = None
n_classes = 2

# Code

class Network:
    def __init__(self, path, train, helper):
        self._helper = helper
        self._train_bool = train

        n_properties = self._helper.n_properties
        n_input = n_properties * w2v_layer_size

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
            "out": tf.Variable(tf.constant(0.0, shape=[n_classes]), name="b_out")
        }

        self._network = self._multilayer_perceptron(self._x, self._weights, self._biases)

    def _multilayer_perceptron(self, _X, _weights, _biases):
        input_layer_drop = tf.nn.dropout(_X, self._input_keep_prob)
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer_drop, _weights["h"]), _biases["b"]))
        hidden_layer_drop = tf.nn.dropout(hidden_layer, self._hidden_keep_prob)
        return tf.matmul(hidden_layer_drop, _weights["out"]) + _biases["out"]

    def train(self, train_dir, model_dir):
        logging.info("Training network using " + train_dir)

        iter = DataSetIterator(self, train_dir, nn_batch_size)

        self._cat_embeddings = Embeddings(iter.cat_lexicon, w2v_layer_size, nn_embed_random_range, True)
        self._slot_embeddings = Embeddings(iter.slot_lexicon, w2v_layer_size, nn_embed_random_range, True)
        self._dist_embeddings = Embeddings(iter.dist_lexicon, w2v_layer_size, nn_embed_random_range, True)
        self._pos_embeddings = Embeddings(iter.pos_lexicon, w2v_layer_size, nn_embed_random_range, True)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._network, self._y))
        regularizers = tf.nn.l2_loss(self._weights["h"]) + tf.nn.l2_loss(self._weights["out"]) + tf.nn.l2_loss(self._biases["b"]) + tf.nn.l2_loss(self._biases["out"])
        cost += nn_l2_reg * regularizers

        optimizer = tf.train.AdagradOptimizer(learning_rate=nn_learning_rate).minimize(cost)
        grads_wrt_input_op = tf.gradients(cost, self._x)[0]

        init = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=0)

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(1, nn_epochs+1):
                logging.info("Training epoch " + str(epoch))

                curr_batch = 1
                sum_cost = 0

                while True:
                    next_batch = iter.next()
                    if not next_batch:
                        break

                    batch_xs, batch_ys, records_in_batch = next_batch

                    logging.info("Training batch " + str(epoch) + "/" + str(curr_batch))

                    _, grads_wrt_input = sess.run([optimizer, grads_wrt_input_op], feed_dict={self._x: batch_xs, self._y: batch_ys, self._input_keep_prob: nn_dropout, self._hidden_keep_prob: nn_dropout})

                    logging.info("Network updated")

                    for i in range(len(records_in_batch)):
                        record = records_in_batch[i]
                        grad_wrt_input = nn_learning_rate * grads_wrt_input[i]
                        record.update_embeddings(grad_wrt_input, w2v_layer_size, self._cat_embeddings, self._slot_embeddings, self._dist_embeddings, self._pos_embeddings)

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

                iter.reset()

            self._serialize(saver, sess, model_dir)

            logging.info("Network training complete")

    def test(self, test_dir, log_file):
        logging.info("Testing network using " + test_dir)

        iter = DataSetIterator(self, test_dir, 0)

        self._cat_embeddings = Embeddings(self._model_dir + "/cat.emb", w2v_layer_size, 0, False)
        self._slot_embeddings = Embeddings(self._model_dir + "/slot.emb", w2v_layer_size, 0, False)
        self._dist_embeddings = Embeddings(self._model_dir + "/dist.emb", w2v_layer_size, 0, False)
        self._pos_embeddings = Embeddings(self._model_dir + "/pos.emb", w2v_layer_size, 0, False)

        saver = tf.train.Saver()

        model_path = self._model_dir + "/model.out"

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            batch_xs, batch_ys, records_in_batch = iter.next()

            logging.info("Number of test examples: " + str(len(records_in_batch)))

            correct_prediction = tf.equal(tf.argmax(self._network,1), tf.argmax(self._y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            y_p = tf.argmax(self._network, 1)
            y_p_raw = tf.split(1, 2, self._network)[1]

            val_accuracy, y_network, y_network_raw = sess.run([accuracy, y_p, y_p_raw], feed_dict={self._x: batch_xs, self._y: batch_ys, self._input_keep_prob: 1.0, self._hidden_keep_prob: 1.0})

            y_true = np.argmax(batch_ys, 1)

            logging.info("Accuracy: " + str(val_accuracy))

            self._evaluate_thresholds(y_true, y_network_raw)

        with open(log_file+".classified1", "w") as out_correct, \
             open(log_file+".classified0", "w") as out_incorrect:

            logging.info("Writing to files")

            for i in range(len(records_in_batch)):
                prediction = y_network[i]

                if prediction >= 0.5:
                    out_correct.write(" ".join(records_in_batch[i].list) + "\n")
                else:
                    out_incorrect.write(" ".join(records_in_batch[i].list) + "\n")

        logging.info("Network testing complete")

    def _evaluate_thresholds(self, y_true, y_network_raw):
        for j in range(5, 10):
            pos_threshold = j * 0.1
            neg_threshold = (10 - j) * 0.1

            self._evaluate_threshold(y_true, y_network_raw, pos_threshold, neg_threshold)

        self._evaluate_threshold(y_true, y_network_raw, 0.8, 0.1)

    def _evaluate_threshold(self, y_true, y_network_raw, pos_threshold, neg_threshold):
        sub_true = list()
        sub_network = list()

        for i in range(len(y_true)):
            # inverse logit
            prediction = math.exp(y_network_raw[i]) / (math.exp(y_network_raw[i]) + 1)

            if prediction >= pos_threshold:
                sub_true.append(y_true[i])
                sub_network.append(1)
            elif prediction <= neg_threshold:
                sub_true.append(y_true[i])
                sub_network.append(0)

        logging.info("Evaluation threshold: " + str(pos_threshold) + ", " + str(neg_threshold))

        sub_true.append(0)
        sub_network.append(0)
        sub_true.append(0)
        sub_network.append(1)
        sub_true.append(1)
        sub_network.append(0)
        sub_true.append(1)
        sub_network.append(1)

        confusion_matrix = sklearn.metrics.confusion_matrix(sub_true, sub_network)
        confusion_matrix -= 1

        logging.info("Examples labeled as 0 classified by model as 0: " + str(confusion_matrix[0][0]))
        logging.info("Examples labeled as 0 classified by model as 1: " + str(confusion_matrix[0][1]))
        logging.info("Examples labeled as 1 classified by model as 0: " + str(confusion_matrix[1][0]))
        logging.info("Examples labeled as 1 classified by model as 1: " + str(confusion_matrix[1][1]))

        logging.info("")

    def _writeUTF(self, string):
        utf8 = string.encode("utf-8")
        length = len(utf8)
        return struct.pack("!H", length) + struct.pack("!" + str(length) + "s", utf8)

    def _serialize(self, saver, sess, model_dir):
        logging.info("Serializing network")
        saver.save(sess, model_dir + "/model.out")

        wh = self._weights["h"].eval().reshape((1,-1), order="F")
        wout = self._weights["out"].eval().reshape((1,-1), order="F")
        bb = self._biases["b"].eval().reshape((1,-1), order="F")
        bout = self._biases["out"].eval().reshape((1,-1), order="F")

        h = np.hstack((wh, wout, bb, bout))

        if sys.byteorder == "little":
            h.byteswap(True)

        r, c = h.shape

        with open(model_dir + "/coeffs", "wb") as coeffs_file:
            coeffs_file.write(struct.pack("!i", 2))
            coeffs_file.write(struct.pack("!i", r))
            coeffs_file.write(struct.pack("!i", c))
            coeffs_file.write(struct.pack("!i", 1))
            coeffs_file.write(struct.pack("!i", 1))
            coeffs_file.write(self._writeUTF("float"))
            coeffs_file.write(self._writeUTF("real"))
            coeffs_file.write(self._writeUTF("HEAP"))
            coeffs_file.write(struct.pack("!i", c))
            coeffs_file.write(self._writeUTF("FLOAT"))

        with open(model_dir + "/coeffs", "ab") as coeffs_file:
            h.tofile(coeffs_file, "")

        logging.info("Serializing embeddings")
        self._cat_embeddings.serialize(model_dir + "/cat.emb")
        self._slot_embeddings.serialize(model_dir + "/slot.emb")
        self._dist_embeddings.serialize(model_dir + "/dist.emb")
        self._pos_embeddings.serialize(model_dir + "/pos.emb")
