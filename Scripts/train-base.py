# This file is the base file for all of my convoluted neural networks

import numpy as np
import tensorflow as tf
from tensorflow import keras
import uuid

print("Loading EMNIST data...")
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
print("Data preprocessing complete.")

# Ask user for kernel size, default to 3 if they just press Enter
kernel_input = input("Enter kernel size (e.g., 2, 3, 4, 5). Default is 3: ")
kernel_size_val = int(kernel_input) if kernel_input else 3
kernel_size = (kernel_size_val, kernel_size_val)

num_classes = len(mapping)

print("Building CNN model...")
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
print("Compiling complete.")

unique_id = uuid.uuid4().hex[:16]
print("Saving model...")
model.save(f"Models/model_{kernel_size[0]}x{kernel_size[1]}_{unique_id}.h5")
print("Model saved successfully.")