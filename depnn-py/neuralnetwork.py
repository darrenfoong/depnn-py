# adapted from https://github.com/aymericdamien/TensorFlow-Examples/

import tensorflow as tf
import wordvectors
import embeddings
import dataset

# Parameters

w2v_layer_size = 50

nn_epochs = 30
nn_batch_size = 128
nn_hidden_layer_size = 200
nn_learning_rate = 1e-2
nn_l2_reg = 1e-8
nn_dropout = 0.5
nn_embed_random_range = 0.01
nn_hard_labels = True

n_input = n_properties * w2v_layer_size
n_classes = 2

# Code

class Network:
    def __init__(self, path, train):
        self.train = train

        if train:
            self.prev_model = path
        else:
            self.model_dir = path

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
        self.network = multilayer_perceptron(self, self.weights, self.biases)

    def multilayer_perceptron(self, _X, _weights, _biases):
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(_X, _weights["h"], _biases["b"]))
        return tf.matmul(hidden_layer, _weights["out"]) + _biases["out"]

    def train(self, deps_dir, model_dir):
        word_vectors = WordVectors(prev_model)

        dataset = Dataset(deps_dir, nn_batch_size)

        cat_embeddings = Embeddings(dataset.cat_lexicon)
        slot_embeddings = Embeddings(dataset.slot_lexicon)
        dist_embeddings = Embeddings(dataset.dist_lexicon)
        pos_embeddings = Embeddings(dataset.pos_lexicon)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=nn_learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(nn_epochs):
                avg_cost = 0
                num_batch = dataset.num_batch()

                for i in range(num_batch):
                    batch_xs, batch_ys = dataset.next_batch()

                    print "Training batch " + str(epoch) + "/" + str(i)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                    avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch

                    print "Cost: " +  avg_cost

                    # TODO get errors and update embeddings

                print "Serializing network"
                # TODO need to create folder!
                saver.save(sess, model_dir + "/epoch " + str(i) + "/model.out")
                cat_embeddings.serialize(model_dir + "/epoch " + str(i) + "/cat.emb")
                slot_embeddings.serialize(model_dir + "/epoch " + str(i) + "/slot.emb")
                dist_embeddings.serialize(model_dir + "/epoch " + str(i) + "/dist.emb")
                pos_embeddings.serialize(model_dir + "/epoch " + str(i) + "/pos.emb")

                dataset.reset()

            saver.save(sess, model_dir + "/model.out")
            cat_embeddings.serialize(model_dir + "/cat.emb")
            slot_embeddings.serialize(model_dir + "/slot.emb")
            dist_embeddings.serialize(model_dir + "/dist.emb")
            pos_embeddings.serialize(model_dir + "/pos.emb")

            print "Network training complete"

    def test(self, deps_dir):
        word_vectors = WordVectors(model_dir + "word2vec.txt")

        dataset = Dataset(deps_dir)

        cat_embeddings = Embeddings(model_dir + "/cat.emb")
        slot_embeddings = Embeddings(model_dir + "/slot.emb")
        dist_embeddings = Embeddings(model_dir + "/dist.emb")
        pos_embeddings = Embeddings(model_dir + "/pos.emb")

        saver = tf.train.Saver()

        model_path = self.model_dir + "/model.out"

        with tf.Session() as sess:
            saver.restore(sess, model_path)

        print "Network testing complete"
