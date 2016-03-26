import numpy as np
import itertools
import os
import random

class Dataset:
    def __init__(self, network, deps_dir, batch_size):
        self.network = network
        self.batch_size = batch_size
        self._correct_deps = list()
        self._incorrect_deps = list()
        self._correct_iter = None
        self._incorrect_iter = None

        deps_file_path = next(os.walk(deps_dir))[2]

        for deps_file_path in deps_file_paths:
            with open(deps_file_path, "r") as deps_file:
                for line in iter(deps_file):
                    line_split = line.split(" ")
                    value = line_split[-1]
                    if value >= 0.5:
                        self._correct_deps.add(line_split)
                    else:
                        self._incorrect_deps.add(line_split)

        num_correct_deps = len(self._correct_deps)
        num_incorrect_deps = len(self._incorrect_deps)
        num_total_deps = num_correct_deps + num_incorrect_deps
        ratio = num_correct_deps / float(num_total_deps)

        if batch_size == 0:
            self._correct_deps_per_batch = num_correct_deps
            self._incorrect_deps_per_batch = num_incorrect_deps
        else:
            self._correct_deps_per_batch = int(ratio * batch_size)
            self._incorrect_deps_per_batch = batch_size - self._correct_deps_per_batch

        reset()

    def reset(self):
        random.shuffle(self._correct_deps)
        random.shuffle(self._incorrect_deps)

        self._correct_iter = iter(self._correct_deps)
        self._incorrect_iter = iter(self._incorrect_deps)

    def next(self):
        deps_in_batch = list()

        for i in range(self._correct_deps_per_batch):
            try:
                deps_in.batch.add(self._correct_iter.next())
            except StopIteration:
                break

        for i in range(self._incorrect_deps_per_batch):
            try:
                deps_in.batch.add(self._incorrect_iter.next())
            except StopIteration:
                break

        if not deps_in_batch:
            return None

        batch_xs = map((lambda dep: network.make_vector(dep[:-1])), deps_in_batch)
        batch_ys = map((lambda dep: dep[-1]), deps_in_batch)

        return (batch_xs, batch_ys)













