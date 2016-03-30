import numpy as np
import re

class Dependency:
    def _init_(self):
        self.list = list()
        self.value = 0.0

    def make_record(self, line, cat_lexicon, slot_lexicon, dist_lexicon, pos_lexicon):
        result = Dependency()

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

        head_vector = network._word_vectors.get(head.lower())
        dependent_vector = network._word_vectors.get(dependent.lower())

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
