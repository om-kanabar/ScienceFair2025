# This file's purpose is to check the data to make sure that it is not coruppted.

print("Loading EMNIST dataset... Please wait.")

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    image = example['image']
    label = example['label']

images_labels = list(train_dataset.take(9))
builder = tfds.builder('emnist/byclass')
names = builder.info.features['label'].names

mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[label_index] = chr(ascii_code)

rows = get_int("Rows (default 3): ", 3)
cols = get_int("Columns (default 3): ", 3)

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

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