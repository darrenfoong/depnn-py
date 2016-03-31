import numpy as np
import itertools
import logging

class WordVectors:
    def __init__(self, path, size_embeddings, unk_string):
        self._unk = 0
        self._unk_string = unk_string

        self._map = dict()
        self._size_embeddings = size_embeddings

        with open(path, "r") as embeddings_file:
            num_embeddings = sum(1 for line in embeddings_file)

        self._embeddings = np.empty(shape=(num_embeddings, self._size_embeddings))

        with open(path, "r") as embeddings_file:
            for line in iter(embeddings_file):
                line = line.replace("\n", "")
                line_split = line.split(" ")
                embedding = line_split[1:]
                self._add(line_split[0], map((lambda s: float(s)), embedding))

    def get(self, key):
        if key in self._map:
            return self._embeddings[self._map[key]]
        else:
            return self._embeddings[self._unk]

    def _add(self, key, embedding):
        current_index = len(self._map)
        self._map[key] = current_index
        self._embeddings[current_index] = embedding/np.linalg.norm(embedding)

        if key == self._unk_string:
            self._unk = current_index
            logging.info("wordVectors has previous UNK: " + key)
            logging.info("Remapping UNK")
