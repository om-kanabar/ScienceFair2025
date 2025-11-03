# This file's purpose is to check the data to make sure that it is not coruppted.

from rich.console import Console

console = Console()

console.print("[bold cyan]Loading EMNIST dataset... Please wait.[/bold cyan]")

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import random
import numpy as np

def get_int(prompt, default):
    user_input = input(prompt)
    return int(user_input) if user_input.strip() != "" else default

source = input("Enter 0 for base EMNIST, 1 for processed balanced dataset: ").strip()
if source == '1':
    data = np.load("Data/emnist-byclass-balanced.npz")
    images = data['images']
    labels = data['labels']
    num_images = get_int("How many images? (default 9): ", 9)
    indices = random.sample(range(len(images)), num_images)
    random_samples = [(images[i], labels[i]) for i in indices]
else:
    train_dataset = tfds.load('emnist/byclass', split='train')
    sample_pool = list(train_dataset.take(100))
    num_images = get_int("How many images? (default 9): ", 9)
    random_samples = random.sample(sample_pool, num_images)

    builder = tfds.builder('emnist/byclass')
    names = builder.info.features['label'].names

    mapping = {}
    with open("Data/emnist-byclass-mapping.txt", "r") as f:
        for line in f:
            label_index, ascii_code = map(int, line.strip().split())
            mapping[label_index] = chr(ascii_code)

console.print("[green]Dataset loaded successfully.[/green]")

rows = get_int("Rows (default 3): ", 3)
cols = get_int("Columns (default 3): ", 3)

fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

console.print("[bold yellow]Displaying sample images...[/bold yellow]")

if source == '1':
    for index, (img, label) in enumerate(random_samples):
        img = img.squeeze()
        img = img / 255.0
        plt.subplot(rows, cols, index + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')
else:
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

console.print("[bold green]Image display complete![/bold green]")