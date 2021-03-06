import numpy as np
import re

class LongDependency:
    def __init__(self):
        self.n_properties = 11
        self.list = None
        self.value = None

    def make_record(self, line, cat_lexicon, slot_lexicon, dist_lexicon, pos_lexicon):
        result = LongDependency()

        # remove count and also the newline, conveniently
        line_split = line.split(" ")[:-1]
        MARKUP_CAT = re.compile(r'\[.*?\]')
        line_split[1] = MARKUP_CAT.sub("", line_split[1])

        result.list = line_split[:-1]
        result.value = float(line_split[-1])

        cat_lexicon.add(result.list[1])
        slot_lexicon.add(result.list[2])
        dist_lexicon.add(result.list[4])
        pos_lexicon.add(result.list[5])
        pos_lexicon.add(result.list[6])
        pos_lexicon.add(result.list[7])
        pos_lexicon.add(result.list[8])
        pos_lexicon.add(result.list[9])
        pos_lexicon.add(result.list[10])

        return result

    def make_vector(self, network):
        head = self.list[0]
        category = self.list[1]
        slot = self.list[2]
        dependent = self.list[3]
        distance = self.list[4]
        head_pos = self.list[5]
        dependent_pos = self.list[6]
        head_left_pos = self.list[7]
        head_right_pos = self.list[8]
        dependent_left_pos = self.list[9]
        dependent_right_pos = self.list[10]

        head_vector = network._word_vectors.get(head)
        dependent_vector = network._word_vectors.get(dependent)

        category_vector = network._cat_embeddings.get(category)
        slot_vector = network._slot_embeddings.get(slot)
        distance_vector = network._dist_embeddings.get(distance)
        head_pos_vector = network._pos_embeddings.get(head_pos)
        dependent_pos_vector = network._pos_embeddings.get(dependent_pos)
        head_left_pos_vector = network._pos_embeddings.get(head_left_pos)
        head_right_pos_vector = network._pos_embeddings.get(head_right_pos)
        dependent_left_pos_vector = network._pos_embeddings.get(dependent_left_pos)
        dependent_right_pos_vector = network._pos_embeddings.get(dependent_right_pos)

        return np.hstack((head_vector, category_vector, slot_vector, dependent_vector, distance_vector, head_pos_vector, dependent_pos_vector, head_left_pos_vector, head_right_pos_vector, dependent_left_pos_vector, dependent_right_pos_vector))

    def update_embeddings(self, grad_wrt_input, w2v_layer_size, cat_embeddings, slot_embeddings, dist_embeddings, pos_embeddings):
        cat_embeddings.update(self.list[1], grad_wrt_input, 1 * w2v_layer_size)
        slot_embeddings.update(self.list[2], grad_wrt_input, 2 * w2v_layer_size)
        dist_embeddings.update(self.list[4], grad_wrt_input, 4 * w2v_layer_size)
        pos_embeddings.update(self.list[5], grad_wrt_input, 5 * w2v_layer_size)
        pos_embeddings.update(self.list[6], grad_wrt_input, 6 * w2v_layer_size)
        pos_embeddings.update(self.list[7], grad_wrt_input, 7 * w2v_layer_size)
        pos_embeddings.update(self.list[8], grad_wrt_input, 8 * w2v_layer_size)
        pos_embeddings.update(self.list[9], grad_wrt_input, 9 * w2v_layer_size)
        pos_embeddings.update(self.list[10], grad_wrt_input, 10 * w2v_layer_size)
