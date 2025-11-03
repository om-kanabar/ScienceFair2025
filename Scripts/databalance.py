# This script balances the neural network's training data.

import tensorflow_datasets as tfds
import numpy as np
import random
from rich.console import Console

console = Console()

console.print("[bold cyan]Starting data balancing script...[/bold cyan]")

# Load the EMNIST ByClass training dataset with labels and images
train_dataset = tfds.load('emnist/byclass', split='train', as_supervised=True)

# Map the labels to actual characters using the provided mapping file
mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code =  map(int, line.split())
        mapping[label_index] = chr(ascii_code)

# Create a dictionary to hold images grouped by their character class
images_by_class = {char: [] for char in mapping.values()}

# Populate the dictionary by iterating through the dataset and grouping images by their character label
for image, label in tfds.as_numpy(train_dataset):
    char = mapping[label]
    images_by_class[char].append(image)

# Define the target number of samples per class for oversampling
target = 20000

# Oversample each class to reach the target number of samples
for char in images_by_class:
    current_count = len(images_by_class[char])
    if current_count < 20000:
        remaining = target - current_count
        images_by_class[char].extend(random.choices(images_by_class[char], k = remaining))
    console.print(f"[yellow]Oversampled class '{char}' to {len(images_by_class[char])} samples.[/yellow]")

console.print("[green]Data balancing script completed successfully![/green]")

# Flatten images and labels
all_images = []
all_labels = []
for char, imgs in images_by_class.items():
    for img in imgs:
        all_images.append(img)
        all_labels.append(char)

# Convert to NumPy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Save to compressed NPZ file
np.savez_compressed('Data/emnist-byclass-balanced.npz', images=all_images, labels=all_labels)
console.print("[bold green]Balanced dataset saved to Data/emnist-byclass-balanced.npz[/bold green]")