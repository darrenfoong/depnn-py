import numpy as np
import itertools

class WordVectors:
    def __init__(self, path, unk_string):
        self._index = dict()
        self.unk_string = unk_string

        with open(path, "r") as embeddings_file:
            num_embeddings = sum(1 for line in embeddings_file)

        self._vectors = np.empty(shape=(num_embeddings, 50))

        with open(path, "r") as embeddings_file:
            for line in iter(embeddings_file):
                line_split = line.split(" ")
                embedding = line_split[1:-1]
                add(line_split[0], map((lambda s: float(s)), embedding))

    def _add(self, entry, vector):
        current_index = len(self._index)
        self._index[entry] = current_index
        self._vectors[current_index] = vector/np.linalg.norm(vector)

    def get(self, entry):
        if entry in self._index:
            return self._vectors[self._index[entry]]
        else:
            return self._vectors[0]
