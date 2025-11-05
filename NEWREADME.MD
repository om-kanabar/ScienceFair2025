# The Effect of Convolutional Kernel Size on Handwritten Character Recognition Accuracy
### By: Om Kanabar

*This project is distributed under a separate license agreement. By using, copying, modifying, or distributing any part of this repository, you agree to the terms and conditions of that license.*

## A few key terms

1. **Neural Network**: A neural network is a computer program that is based on the human brain. This works by learning patterns from data. A neural network modifies its connections to learn. A neural network can recognize images or understand speech, or even make a prediction. Each hidden layer in the neural network will perform a mathematical operation to help it make its prediction. (See Fig 1)

![Neural Network Visualization](Assets/NeuralNetworkVisualization.png "Neural Network Visualization")
---
Fig. 1 Neural Network Visualization.

---

2. **Convolutional Neural Network (CNN)**: A type of neural network that is designed to for images. It is preferred for image tasks due to it processing images using layers that detect patterns like edges, shapes, and textures.

3. **Kernel**: A small array of numbers that is also known as a filter, it moves across an image in a CNN in order to spot patterns or edges, or shapes. This kernel’s size will determine just how much of an image is processed at one given point in order to detect detailed or large features.

## A bit about this project

This project deals with how the kernel size changes affect the accuracy of CNN's which are made for recognizing handwritten characters using the EMINST byclass dataset. This experiment intends to find out how different kernel sizes influence learning efficiency and classification performances since several models will be trained with different kernel sizes.

## Objective/Purpose
The goal of this project is to investigate how convolutional kernel size affects the accuracy and learning efficiency of CNNs trained on handwritten character data

## Purpose Statment

##### How does convolutional kernel size affect handwritten charecter recognition accuracy?

## Hypothesis
##### If the kernel size of a convolutional neural network is 4x4, it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details.