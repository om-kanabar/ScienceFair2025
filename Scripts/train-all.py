# This script automates running train-model.py 12 times:
# 3 runs each for kernel sizes 2x2, 3x3 (control), 4x4, and 5x5.

import subprocess
import time
from rich.console import Console
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
train_model_path = os.path.join(script_dir, "train-model.py")

console = Console()

kernel_sizes = [2, 3, 4, 5]
runs_per_size = 3

console.print("[bold cyan]Starting automated training sequence...[/bold cyan]\n")

for kernel in kernel_sizes:
    for run in range(1, runs_per_size + 1):
        console.print(f"[yellow]Running model {run}/3 with kernel size {kernel}x{kernel}...[/yellow]")
        
        # Run train-model.py and pass the kernel size as input
        process = subprocess.run(
            ["python3", train_model_path],
            input=str(kernel) + "\n",  # feed kernel size to input()
            text=True
        )

        if process.returncode == 0:
            console.print(f"[green] Completed: Kernel {kernel}x{kernel}, Run {run}/3[/green]\n")
        else:
            console.print(f"[bold red] Failed: Kernel {kernel}x{kernel}, Run {run}/3[/red]\n")

        # Pause briefly between runs
        time.sleep(5)

console.print("[bold green]All 12 training runs completed successfully![/bold green]")