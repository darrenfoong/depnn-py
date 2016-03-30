import numpy as np
import itertools
import os
import random
import logging

class Dataset:
    def __init__(self, network, deps_dir, batch_size):
        self._network = network
        self._batch_size = batch_size
        self._helper = network._helper
        self._correct_deps = list()
        self._incorrect_deps = list()
        self._correct_iter = None
        self._incorrect_iter = None

        self.cat_lexicon = set()
        self.slot_lexicon = set()
        self.dist_lexicon = set()
        self.pos_lexicon = set()

        deps_file_paths = next(os.walk(deps_dir))[2]

        for deps_file_path in deps_file_paths:
            with open(deps_dir + "/" + deps_file_path, "r") as deps_file:
                for line in iter(deps_file):
                    record = self._helper.make_record(line, self.cat_lexicon, self.slot_lexicon, self.dist_lexicon, self.pos_lexicon)

                    if record.value >= 0.5:
                        self._correct_deps.append(record)
                    else:
                        self._incorrect_deps.append(record)

        num_correct_deps = len(self._correct_deps)
        num_incorrect_deps = len(self._incorrect_deps)
        num_total_deps = num_correct_deps + num_incorrect_deps
        ratio = num_correct_deps / float(num_total_deps)

        if self._batch_size == 0:
            self._correct_deps_per_batch = num_correct_deps
            self._incorrect_deps_per_batch = num_incorrect_deps
        else:
            self._correct_deps_per_batch = int(ratio * self._batch_size)
            self._incorrect_deps_per_batch = self._batch_size - self._correct_deps_per_batch

        logging.info("Number of correct deps: " + str(num_correct_deps))
        logging.info("Number of incorrect deps: " + str(num_incorrect_deps))
        logging.info("Number of correct deps per batch: " + str(self._correct_deps_per_batch))
        logging.info("Number of incorrect deps per batch: " + str(self._incorrect_deps_per_batch))
        logging.info("All deps read")

        self.reset()

    def reset(self):
        random.shuffle(self._correct_deps)
        random.shuffle(self._incorrect_deps)

        self._correct_iter = iter(self._correct_deps)
        self._incorrect_iter = iter(self._incorrect_deps)

    def next(self):
        deps_in_batch = list()

        for i in range(self._correct_deps_per_batch):
            try:
                deps_in_batch.append(self._correct_iter.next())
            except StopIteration:
                break

        for i in range(self._incorrect_deps_per_batch):
            try:
                deps_in_batch.append(self._incorrect_iter.next())
            except StopIteration:
                break

        if not deps_in_batch:
            return None

        batch_xs = map((lambda dep: dep.make_vector(self._network)), deps_in_batch)
        batch_ys = map(self._make_labels, deps_in_batch)

        return (batch_xs, batch_ys, deps_in_batch)

    def _make_labels(self, dep):
        if float(dep.value) == 0.0:
            return [1, 0]
        else:
            return [0, 1]
