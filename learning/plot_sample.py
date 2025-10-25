import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

train_dataset = tfds.load('emnist/mixed', split='train')

for example in train_dataset.take(1):
    image = example['image']      # tf.Tensor, shape (28,28,1)
    label = example['label']      # integer 0–35

image = image.numpy().squeeze()  # shape (28,28)
image = image / 255.0

images_labels = list(train_dataset.take(9))  # list of 9 dictionaries
