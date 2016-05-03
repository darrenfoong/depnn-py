import numpy as np
import re

class Feature:
    def __init__(self):
        self.n_properties = 9
        self.list = None
        self.value = None

    def make_record(self, line, cat_lexicon, slot_lexicon, dist_lexicon, pos_lexicon):
        result = Feature()

        # remove count and also the newline, conveniently
        line_split = line.split(" ")[:-1]
        #MARKUP_CAT = re.compile(r'\[.*?\]')
        #line_split[1] = MARKUP_CAT.sub("", line_split[1])

        result.list = line_split[:-1]
        result.value = float(line_split[-1])

        cat_lexicon.add(result.list[0])
        cat_lexicon.add(result.list[1])
        cat_lexicon.add(result.list[2])
        pos_lexicon.add(result.list[6])
        pos_lexicon.add(result.list[7])
        pos_lexicon.add(result.list[8])

        return result

    def make_vector(self, network):
        top_cat = self.list[0]
        left_cat = self.list[1]
        right_cat = self.list[2]
        top_cat_word = self.list[3]
        left_cat_word = self.list[4]
        right_cat_word = self.list[5]
        top_cat_pos = self.list[6]
        left_cat_pos = self.list[7]
        right_cat_pos = self.list[8]

        top_cat_vector = network._cat_embeddings.get(top_cat)
        left_cat_vector = network._cat_embeddings.get(left_cat)
        right_cat_vector = network._cat_embeddings.get(right_cat)
        top_cat_word_vector = network._word_vectors.get(top_cat_word)
        left_cat_word_vector = network._word_vectors.get(left_cat_word)
        right_cat_word_vector = network._word_vectors.get(right_cat_word)
        top_cat_pos_vector = network._pos_embeddings.get(top_cat_pos)
        left_cat_pos_vector = network._pos_embeddings.get(left_cat_pos)
        right_cat_pos_vector = network._pos_embeddings.get(right_cat_pos)

        return np.hstack((top_cat_vector, left_cat_vector, right_cat_vector, top_cat_word_vector, left_cat_word_vector, right_cat_word_vector, top_cat_pos_vector, left_cat_pos_vector, right_cat_pos_vector))

    def update_embeddings(self, grad_wrt_input, w2v_layer_size, cat_embeddings, slot_embeddings, dist_embeddings, pos_embeddings):
        cat_embeddings.update(self.list[0], grad_wrt_input, 0 * w2v_layer_size)
        cat_embeddings.update(self.list[1], grad_wrt_input, 1 * w2v_layer_size)
        cat_embeddings.update(self.list[2], grad_wrt_input, 2 * w2v_layer_size)
        pos_embeddings.update(self.list[6], grad_wrt_input, 6 * w2v_layer_size)
        pos_embeddings.update(self.list[7], grad_wrt_input, 7 * w2v_layer_size)
        pos_embeddings.update(self.list[8], grad_wrt_input, 8 * w2v_layer_size)
