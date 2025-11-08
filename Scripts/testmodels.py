# This script tests all of the models

from rich.console import Console

console = Console()

console.print("[bold cyan]Importing packages... \n")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import os
import csv

console.print("[green]Imported packages\n")

console.print("[bold cyan]Fetching models\n")

models_dir = "Models"
model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model = keras.models.load_model(model_path)

console.print("[green]Models fetched \n")
console.print("[bold cyan]Loading and converting EMNIST test data...\n")

data = tfds.load('emnist/byclass', split='test', as_supervised=True)

# Convert the tf.data.Dataset into NumPy arrays
images = []
labels = []

for image, label in tfds.as_numpy(data):
    images.append(image)
    labels.append(label)

x = np.array(images)
y = np.array(labels)

console.print(f"[green]Loaded {len(x)} test samples.\n")

console.print("[green]Data Loaded\n")

x = x / 255.0
x = x.reshape(-1, 28, 28, 1)

console.print("[bold cyan]Evaluating models...\n")

results = {}

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    console.print(f"[yellow]Evaluating {model_file} 3 times...[/yellow]")
    
    run_accuracies = []
    run_losses = []
    for run in range(1, 4):
        model = keras.models.load_model(model_path)
        loss, accuracy = model.evaluate(x, y, verbose=0)
        run_accuracies.append(accuracy)
        run_losses.append(loss)
        console.print(f"[green]Run {run} — Accuracy: {accuracy:.4f}, Loss: {loss:.4f}[/green]")
    avg_accuracy = sum(run_accuracies) / 3
    avg_loss = sum(run_losses) / 3
    console.print(f"[bold green]Average — Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}[/bold green]\n")
    results[model_file] = {
        "accuracies": run_accuracies,
        "losses": run_losses,
        "avg_accuracy": avg_accuracy,
        "avg_loss": avg_loss
    }

with open("Results/model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model Name", "Kernel Size", "Run 1 Accuracy", "Run 1 Loss", "Run 2 Accuracy", "Run 2 Loss", "Run 3 Accuracy", "Run 3 Loss", "Average Accuracy", "Average Loss"])

    for model_file in model_files:
        # Extract kernel size from filename
        # Example filename: model_3x3_abcdef1234567890.keras
        parts = model_file.split('_')
        if len(parts) >= 3:
            kernel_size = parts[1]
        else:
            kernel_size = "Unknown"
        
        print(f"Extracted kernel size for {model_file}: {kernel_size}")
        run_acc = results[model_file]["accuracies"]
        run_loss = results[model_file]["losses"]
        avg_acc = results[model_file]["avg_accuracy"]
        avg_loss = results[model_file]["avg_loss"]
        writer.writerow([model_file, kernel_size, run_acc[0], run_loss[0], run_acc[1], run_loss[1], run_acc[2], run_loss[2], avg_acc, avg_loss])

console.print("[bold green]All results saved to Results/model_results.csv[/bold green]")