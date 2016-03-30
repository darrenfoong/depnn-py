import numpy as np
import re

class TransDependency:
    def __init__(self):
        self.n_properties = 5
        self.list = None
        self.value = None

    def make_record(self, line, cat_lexicon, slot_lexicon, dist_lexicon, pos_lexicon):
        result = TransDependency()

        # remove count and also the newline, conveniently
        line_split = line.split(" ")[:-1]
        MARKUP_CAT = re.compile(r'\[.*?\]')
        line_split[1] = MARKUP_CAT.sub("", line_split[1])

        if line_split[1] != "(S\\NP)/NP" or line_split[2] != "2":
            return None

        result.list = list()

        result.list.append(line_split[0])
        result.list.append(line_split[3])
        result.list.append(line_split[4])
        result.list.append(line_split[5])
        result.list.append(line_split[6])
        result.value = float(line_split[-1])

        dist_lexicon.add(result.list[2])
        pos_lexicon.add(result.list[3])
        pos_lexicon.add(result.list[4])

        return result

    def make_vector(self, network):
        head = self.list[0]
        dependent = self.list[1]
        distance = self.list[2]
        head_pos = self.list[3]
        dependent_pos = self.list[4]

        head_vector = network._word_vectors.get(head.lower())
        dependent_vector = network._word_vectors.get(dependent.lower())

        distance_vector = network._dist_embeddings.get(distance)
        head_pos_vector = network._pos_embeddings.get(head_pos)
        dependent_pos_vector = network._pos_embeddings.get(dependent_pos)

        return np.hstack((head_vector, dependent_vector, distance_vector, head_pos_vector, dependent_pos_vector))

    def update_embeddings(self, grad_wrt_input, w2v_layer_size, cat_embeddings, slot_embeddings, dist_embeddings, pos_embeddings):
        dist_embeddings.update(self.list[2], grad_wrt_input, 2 * w2v_layer_size)
        pos_embeddings.update(self.list[3], grad_wrt_input, 3 * w2v_layer_size)
        pos_embeddings.update(self.list[4], grad_wrt_input, 4 * w2v_layer_size)
