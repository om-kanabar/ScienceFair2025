# This script builds, compiles, and trains a CNN on the EMNIST dataset.

# Import required libraries
from rich.console import Console

console = Console()

console.print("[bold cyan]Importing packages...")

import numpy as np
import tensorflow as tf
from tensorflow import keras
import uuid
from sklearn.model_selection import train_test_split
import os
import random
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_path = os.path.join(project_root, "Data", "emnist-byclass-balanced.npz")
mapping_path = os.path.join(project_root, "Data", "emnist-byclass-mapping.txt")
models_dir = os.path.join(project_root, "Models")

console.print("[green]Packages imported successfully")


# Load the preprocessed EMNIST balanced dataset
console.print("[bold cyan]Loading balanced EMNIST dataset...[/bold cyan]")

data = np.load(data_path)
train_images = data["images"]
train_labels = data["labels"]

console.print(f"[green]Loaded dataset with {len(train_images)} samples.[/green]")

x = data["images"]
y = data["labels"]


# Load label mapping (character to numeric index)
mapping = {}
with open(mapping_path, "r") as f:
    for line in f:
        label_index, ascii_code = map(int, line.strip().split())
        mapping[chr(ascii_code)] = label_index


# Normalize pixel values and reshape images for CNN input
x = x / 255.0
x = x.reshape(-1, 28, 28, 1)
console.print("[green]Data preprocessing complete.[/green]")


# Get kernel size input from the user
kernel_input = input("Enter kernel size (e.g., 2, 3, 4, 5). Default is 3: ")
kernel_size_val = int(kernel_input) if kernel_input else 3
kernel_size = (kernel_size_val, kernel_size_val)

num_classes = len(mapping)


# Define CNN model structure
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


# Compile the model with Adam optimizer and appropriate loss function
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
console.print("[green]Compiling complete.[/green]")


unique_id = uuid.uuid4().hex[:16]  # Generate a unique 16-character identifier for model naming

train_images = train_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)

# Split dataset into training and validation sets for evaluation
console.print("[bold cyan]Splitting data into training and validation sets...[/bold cyan]")

x_train, x_val, y_train, y_val = train_test_split(
    train_images,
    train_labels,
    test_size=0.1,
    random_state=random.randint(1,50)
)
console.print(f"[green]Training samples: {len(x_train)}, Validation samples: {len(x_val)}[/green]")

# Augment training data by adding Gaussian noise
console.print("[bold cyan]Augmenting training data with Gaussian noise...[/bold cyan]")
noise_factor = 0.1
# Generate Gaussian noise
noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
# Create noisy images by adding noise and clipping to [0,1]
x_train_noisy = np.clip(x_train + noise, 0., 1.)
# Concatenate original and noisy images
x_train_augmented = np.concatenate((x_train, x_train_noisy), axis=0)
# Duplicate labels for noisy images
y_train_augmented = np.concatenate((y_train, y_train), axis=0)

console.print(f"[green]Training data augmented: {len(x_train_augmented)} samples total.[/green]")

console.print("[bold cyan]Training started...[/bold cyan]")

# Add early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # what to watch
    patience=2,              # stop if no improvement for 3 epochs
    restore_best_weights=True  # revert to the best model
)

# Trains the neural network

history = model.fit(
    x_train_augmented,
    y_train_augmented,
    validation_data=(x_val, y_val),
    epochs=15,            # maximum number of epochs
    batch_size=64,
    callbacks=[early_stopping]
)

console.print("[bold green]Training complete![/bold green]")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model.save(os.path.join(models_dir, f"model_{kernel_size[0]}x{kernel_size[1]}_{unique_id}.keras"))
console.print(f"[bold green]Model saved as model_{kernel_size[0]}x{kernel_size[1]}_{unique_id}.keras[/bold green]")