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
import cv2

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

# Apply Gaussian noise to a subset of test images
num_samples = x.shape[0]
num_noisy = int(num_samples * 0.35)  # 35% of samples
indices = np.random.choice(num_samples, num_noisy, replace=False)
noise = np.random.normal(loc=0.0, scale=0.15, size=x.shape)
x_noisy = np.copy(x)
x_noisy[indices] += noise[indices]
x_noisy = np.clip(x_noisy, 0.0, 1.0)
x = x_noisy

console.print("[bold magenta]Note: Gaussian noise is added to help simulate real world.")

console.print(f"[bold cyan]Applied Gaussian noise to {num_noisy} test samples (~35%).\n")

console.print("[bold cyan]Evaluating models...\n")

results = {}

x_clean = np.copy(x)

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    parts = model_file.split('_')
    if len(parts) >= 3:
        kernel_size = parts[1]
    else:
        kernel_size = "Unknown"

    # --- Frequency-domain robustness testing (per-image noise) ---
    # Instead of choosing noise by kernel, compute each image's high-frequency
    # strength (Laplacian) and scale the per-image Gaussian noise by that value.
    # This produces a realistic test condition (stronger noise on high-frequency
    # images).

    console.print(f"[yellow]Evaluating {model_file} 3 times...[/yellow]")
    
    run_accuracies = []
    run_losses = []
    for run in range(1, 4):
        # Start with clean test set
        x_run = np.copy(x_clean)

        # Randomly select indices to perturb
        num_noisy = int(x_run.shape[0] * 0.35)
        indices = np.random.choice(x_run.shape[0], num_noisy, replace=False)

        # Compute per-image Laplacian-based stddev for these indices
        hf_vals = np.zeros(num_noisy, dtype=np.float32)
        for i, idx in enumerate(indices):
            img = (x_run[idx].squeeze() * 255.0).astype(np.uint8)
            lap = cv2.Laplacian(img, cv2.CV_64F)
            hf_vals[i] = np.mean(np.abs(lap))
        hf_norm = (hf_vals - hf_vals.min()) / (hf_vals.max() - hf_vals.min() + 1e-8)

        base_noise = 0.05
        scale_factor = 0.8
        per_image_std = base_noise * (1.0 + scale_factor * hf_norm)

        noise_selected = np.random.normal(loc=0.0, scale=1.0, size=(num_noisy, 28, 28, 1)).astype(np.float32)
        noise_selected *= per_image_std[:, None, None, None]

        x_run[indices] += noise_selected
        x_run = np.clip(x_run, 0.0, 1.0)

        model = keras.models.load_model(model_path)
        loss, accuracy = model.evaluate(x_run, y, verbose=0)
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
