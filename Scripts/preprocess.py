# This file preprocess the data for the main experiment.

import tensorflow_datasets as tfds
import collections

train_dataset = tfds.load('emnist/byclass', split='train', as_supervised=True)
test_dataset = tfds.load('emnist/byclass', split='test', as_supervised=True)

mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[label_index] = chr(ascii_code)

counts = collections.Counter()
for image, label in tfds.as_numpy(train_dataset):
    counts[mapping[label]] += 1

# Nicely formatted counts, sorted by character
for char, count in sorted(counts.items()):
    print(f"{char}: {count}")