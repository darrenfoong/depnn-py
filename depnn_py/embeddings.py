import numpy as np
import itertools
import logging

class Embeddings:
    def __init__(self, lexicon, size_embeddings, random_range, train):
        self._unk = 0
        self._unk_string = "_UNK_"

        self._map = dict()
        self._size_embeddings = size_embeddings
        self._random_range = float(random_range)

        if train:
            self._embeddings = np.empty(shape=(len(lexicon)+1, self._size_embeddings))

            self._add(self._unk_string, self._random_embedding())

            for key in lexicon:
                self._add(key, self._random_embedding())
        else:
            with open(lexicon, "r") as embeddings_file:
                num_embeddings = sum(1 for line in embeddings_file)

            self._embeddings = np.empty(shape=(num_embeddings, self._size_embeddings))

            with open(lexicon, "r") as embeddings_file:
                for line in iter(embeddings_file):
                    line = line.replace("\n", "")
                    line_split = line.split(" ")
                    embedding = line_split[1:]
                    self._add(line_split[0], map((lambda s: float(s)), embedding))

    def serialize(self, path):
        with open(path, "w") as embeddings_file:
            for key, index in self._map.iteritems():
                embedding = self._embeddings[index]
                output = " ".join(map((lambda x: str(x)), embedding))
                embeddings_file.write(key + " " + output + "\n")

    def _random_embedding(self):
        return (np.random.rand(1, self._size_embeddings) * 2 * self._random_range) - self._random_range

    def get(self, key):
        if key in self._map:
            return self._embeddings[self._map[key]]
        else:
            return self._embeddings[self._unk]

    def _add(self, key, embedding):
        current_index = len(self._map)
        self._map[key] = current_index
        self._embeddings[current_index] = embedding
        #self._embeddings[current_index] = embedding/np.linalg.norm(embedding)

        if key == self._unk_string:
            self._unk = current_index
            logging.info("Remapping UNK")

    def update(self, key, embedding, offset):
        sub_embedding = embedding[offset:(offset+self._size_embeddings)]

        if key in self._map:
            self._embeddings[self._map[key]] -= sub_embedding
        else:
            self._embeddings[self._unk] -= sub_embedding
