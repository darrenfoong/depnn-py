import numpy as np
import itertools

class Embeddings:
    def __init__(self, lexicon, train, random_range=0.01):
        self._index = dict()
        self._random_range = float(random_range)

        if train:
            self._vectors = np.empty(shape=(len(lexicon)+1, 50))

            self._add("_UNK_", self._random_vector())

            for entry in lexicon:
                self._add(entry, self._random_vector())
        else:
            with open(lexicon, "r") as embeddings_file:
                num_embeddings = sum(1 for line in embeddings_file)

            self._vectors = np.empty(shape=(num_embeddings, 50))

            with open(lexicon, "r") as embeddings_file:
                for line in iter(embeddings_file):
                    # TODO possible newline bug here
                    line_split = line.split(" ")
                    embedding = line_split[1:]
                    self._add(line_split[0], map((lambda s: float(s)), embedding))

    def _random_vector(self):
        return (np.random.rand(1, 50) * 2 * self._random_range) - self._random_range

    def _add(self, entry, vector):
        current_index = len(self._index)
        self._index[entry] = current_index
        self._vectors[current_index] = vector
        #self._vectors[current_index] = vector/np.linalg.norm(vector)

    def get(self, entry):
        if entry in self._index:
            return self._vectors[self._index[entry]]
        else:
            return self._vectors[0]

    def update(self, entry, vector, learning_rate):
        if entry in self._index:
            current_vector = self._vectors[self._index[entry]]
            self._vectors[self._index[entry]] = current_vector - (learning_rate * vector)
        else:
            current_vector = self._vectors[0]
            self._vectors[0] = current_vector - (learning_rate * vector)

    def serialize(self, path):
        with open(path, "w") as embeddings_file:
            for entry, index in self._index.iteritems():
                vector = self._vectors[index]
                output = " ".join(map((lambda x: str(x)), vector))
                embeddings_file.write(entry + " " + output + "\n")
