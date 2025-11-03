# This file is deprecated due to it's functionality being replaced by train-model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import uuid
from rich.console import Console

console = Console()

console.print("[bold cyan]Loading EMNIST data...[/bold cyan]")
data = np.load("Data/emnist-byclass-balanced.npz")
x = data["images"]
y = data["labels"]

mapping = {}
with open("Data/emnist-byclass-mapping.txt", "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[chr(ascii_code)] = label_index

x = x / 255.0
x = x.reshape(-1, 28, 28, 1)
console.print("[green]Data preprocessing complete.[/green]")

# Ask user for kernel size, default to 3 if they just press Enter
kernel_input = input("Enter kernel size (e.g., 2, 3, 4, 5). Default is 3: ")
kernel_size_val = int(kernel_input) if kernel_input else 3
kernel_size = (kernel_size_val, kernel_size_val)

num_classes = len(mapping)

console.print("[bold cyan]Building CNN model...[/bold cyan]")
# The actual CNN

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=kernel_size, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=kernel_size, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
console.print("[green]Compiling complete.[/green]")

unique_id = uuid.uuid4().hex[:16]
console.print("[bold yellow]Saving model...[/bold yellow]")
model.save(f"Models/model_{kernel_size[0]}x{kernel_size[1]}_{unique_id}.h5")
console.print("[bold green]Model saved successfully.[/bold green]")