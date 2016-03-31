import numpy as np
import itertools
import os
import random
import logging

class Dataset:
    def __init__(self, network, records_dir, batch_size):
        self._network = network
        self._batch_size = batch_size
        self._helper = network._helper
        self._correct_records = list()
        self._incorrect_records = list()
        self._correct_iter = None
        self._incorrect_iter = None

        self.cat_lexicon = set()
        self.slot_lexicon = set()
        self.dist_lexicon = set()
        self.pos_lexicon = set()

        records_file_paths = next(os.walk(records_dir))[2]

        for records_file_path in records_file_paths:
            with open(records_dir + "/" + records_file_path, "r") as records_file:
                for line in iter(records_file):
                    record = self._helper.make_record(line, self.cat_lexicon, self.slot_lexicon, self.dist_lexicon, self.pos_lexicon)

                    if not record:
                        continue

                    if record.value >= 0.5:
                        self._correct_records.append(record)
                    else:
                        self._incorrect_records.append(record)

        num_correct_records = len(self._correct_records)
        num_incorrect_records = len(self._incorrect_records)
        num_total_records = num_correct_records + num_incorrect_records
        ratio = num_correct_records / float(num_total_records)

        if self._batch_size == 0:
            self._correct_records_per_batch = num_correct_records
            self._incorrect_records_per_batch = num_incorrect_records
        else:
            self._correct_records_per_batch = int(ratio * self._batch_size)
            self._incorrect_records_per_batch = self._batch_size - self._correct_records_per_batch

        logging.info("Number of correct records: " + str(num_correct_records))
        logging.info("Number of incorrect records: " + str(num_incorrect_records))
        logging.info("Number of correct records per batch: " + str(self._correct_records_per_batch))
        logging.info("Number of incorrect records per batch: " + str(self._incorrect_records_per_batch))
        logging.info("All records read")

        self.reset()

    def reset(self):
        random.shuffle(self._correct_records)
        random.shuffle(self._incorrect_records)

        self._correct_iter = iter(self._correct_records)
        self._incorrect_iter = iter(self._incorrect_records)

    def next(self):
        records_in_batch = list()

        for i in range(self._correct_records_per_batch):
            try:
                records_in_batch.append(self._correct_iter.next())
            except StopIteration:
                break

        for i in range(self._incorrect_records_per_batch):
            try:
                records_in_batch.append(self._incorrect_iter.next())
            except StopIteration:
                break

        if not records_in_batch:
            return None

        batch_xs = map((lambda record: record.make_vector(self._network)), records_in_batch)
        batch_ys = map(self._make_labels, records_in_batch)

        return (batch_xs, batch_ys, records_in_batch)

    def _make_labels(self, record):
        if float(record.value) == 0.0:
            return [1, 0]
        else:
            return [0, 1]
