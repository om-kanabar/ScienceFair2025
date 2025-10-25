import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import random

def get_int(prompt, default):
    user_input = input(prompt)
    return int(user_input) if user_input.strip() != "" else default

train_dataset = tfds.load('emnist/byclass', split='train')
sample_pool = list(train_dataset.take(100))
num_images = get_int("How many images? (default 9): ", 9)
random_samples = random.sample(sample_pool, num_images)

for example in train_dataset.take(1):
    image = example['image']      # tf.Tensor, shape (28,28,1)
    label = example['label']      # integer 0–35

images_labels = list(train_dataset.take(9))  # list of 9 dictionaries
builder = tfds.builder('emnist/byclass')
names = builder.info.features['label'].names

mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[label_index] = chr(ascii_code)

rows = get_int("Rows (default 3): ", 3)
cols = get_int("Columns (default 3): ", 3)

for index, example in enumerate(random_samples):
    img = example['image'].numpy().squeeze()
    img = img / 255.0
    plt.subplot(rows, cols, index + 1)
    plt.imshow(img, cmap='gray')
    label_value = example['label'].numpy()
    label_name = mapping[label_value]
    plt.title(label_name)
    plt.axis('off')

plt.tight_layout(pad=1.5)
plt.show()

# EMNIST stores labels as category indices.
# TensorFlow Datasets already provides the official human-readable mapping in builder.info.features['label'].names.
# Instead of computing ASCII codes manually, I directly index into that list, which guarantees that the label displayed matches the dataset specification.