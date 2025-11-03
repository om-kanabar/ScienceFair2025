# This file preprocess the data for the main experiment.

import tensorflow_datasets as tfds
import collections
import numpy as np
from rich.console import Console

console = Console()

source = input("Enter 0 for base EMNIST, 1 for processed balanced dataset emnist-byclass-balanced.npz: ")

mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[label_index] = chr(ascii_code)

counts = collections.Counter()

if source == '0':
    console.print("[bold cyan]Using base EMNIST dataset.[/bold cyan]")
    train_dataset = tfds.load('emnist/byclass', split='train', as_supervised=True)
    for image, label in tfds.as_numpy(train_dataset):
        counts[mapping[label]] += 1
elif source == '1':
    console.print("[bold cyan]Using processed balanced dataset emnist-byclass-balanced.npz.[/bold cyan]")
    data = np.load("Data/emnist-byclass-balanced.npz")
    labels = data['labels']
    for label in labels:
        counts[label] += 1
else:
    console.print("[bold red]Invalid input. Please enter 0 or 1.[/bold red]")

console.print("[green]Data counting complete! Displaying results...[/green]")

# Nicely formatted counts, sorted by character
for char, count in sorted(counts.items()):
    console.print(f"[yellow]{char}[/yellow]: [bold white]{count}[/bold white]")
