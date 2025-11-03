# Visualizing how different optimizers move down a loss curve
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define a simple 1D "loss function"
def loss_function(x):
    return x ** 2 + 5 * np.sin(x)

# Compute the gradient (derivative)
def gradient(x):
    return 2 * x + 5 * np.cos(x)

# Training settings
initial_x = 5.0
steps = 1000
learning_rate = 0.1

# TensorFlow optimizers to compare
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=learning_rate),
    "Momentum": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.95),
}

# Prepare the x-axis (for plotting the loss surface)
x_vals = np.linspace(-6, 6, 250)
y_vals = loss_function(x_vals)

# Collect paths for combined plotting
paths = {}

# Initialize subplots for individual optimizer visualization
fig, axes = plt.subplots(len(optimizers), 1, figsize=(8, 4 * len(optimizers)), sharex=True, sharey=True)

if len(optimizers) == 1:
    axes = [axes]

for ax, (name, opt) in zip(axes, optimizers.items()):
    x = tf.Variable(initial_x)
    path = [float(x)]
    for step in range(steps):
        with tf.GradientTape() as tape:
            loss = loss_function(x)
        grads = tape.gradient(loss, [x])
        opt.apply_gradients(zip(grads, [x]))
        path.append(float(x))
    paths[name] = path

    ax.plot(x_vals, y_vals, 'k--', label='Loss function')
    ax.plot(path, loss_function(np.array(path)), marker='o', label=name)
    ax.set_title(f"{name} Descent Path")
    ax.set_xlabel("Parameter value (x)")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-6, 6)
    ax.set_ylim(min(y_vals), max(y_vals))

plt.tight_layout()
plt.show()

# Combined plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'k--', label='Loss function')
for name, path in paths.items():
    plt.plot(path, loss_function(np.array(path)), marker='o', label=name)

plt.title("Optimizer Behavior Comparison")
plt.xlabel("Parameter value (x)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(min(y_vals), max(y_vals))
plt.show()

# Plot loss vs. training step for each optimizer
plt.figure(figsize=(10, 6))
for name, path in paths.items():
    losses = loss_function(np.array(path))
    plt.plot(range(len(path)), losses, marker='o', label=name)
plt.title("Loss vs. Training Step for Each Optimizer")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()