# This script converts all legacy .h5 models in /Models to the new .keras format.

import os
from tensorflow import keras
from rich.console import Console

console = Console()

models_dir = "Models"

console.print("[bold cyan]Converting all .h5 models to .keras format...[/bold cyan]")

# Iterate through the Models folder
for filename in os.listdir(models_dir):
    if filename.endswith(".h5"):
        old_path = os.path.join(models_dir, filename)
        new_path = os.path.splitext(old_path)[0] + ".keras"

        console.print(f"[yellow]Converting {filename} → {os.path.basename(new_path)}...[/yellow]")

        # Load and re-save in new format
        model = keras.models.load_model(old_path)
        model.save(new_path)

        console.print(f"[green] Converted: {filename}[/green]")

console.print("[bold green]All conversions complete![/bold green]")