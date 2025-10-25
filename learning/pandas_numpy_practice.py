
# In this learning experience I learned how to graph 3 pixels in numpy (I can do more)
# I also experimented with label names and wether or not they could be words with
# special charecters or just numbers.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("learning/tiny_data.csv")
images = data[['pixel1', 'pixel2', 'pixel3']].to_numpy()
images = images / 255.0
labels = data['label'].to_numpy()

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].reshape(1,3), cmap="gray")
    plt.title(f"Label = {labels[i]}")

plt.show()