import numpy as np
import itertools
import logging

class WordVectors:
    def __init__(self, path, w2v_layer_size, unk_string):
        self._index = dict()
        self._w2v_layer_size = w2v_layer_size
        self._unk = 0
        self._unk_string = unk_string

        with open(path, "r") as embeddings_file:
            num_embeddings = sum(1 for line in embeddings_file)

        self._vectors = np.empty(shape=(num_embeddings, self._w2v_layer_size))

        with open(path, "r") as embeddings_file:
            for line in iter(embeddings_file):
                line = line.replace("\n", "")
                line_split = line.split(" ")
                embedding = line_split[1:]
                self._add(line_split[0], map((lambda s: float(s)), embedding))

        logging.info("Number of words: " + str(num_embeddings))

    def _add(self, entry, vector):
        current_index = len(self._index)
        self._index[entry] = current_index
        self._vectors[current_index] = vector/np.linalg.norm(vector)

        if entry == self._unk_string:
            self._unk = current_index
            logging.info("wordVectors has previous UNK: " + entry)
            logging.info("Remapping UNK")

    def get(self, entry):
        if entry in self._index:
            return self._vectors[self._index[entry]]
        else:
            return self._vectors[self._unk]
