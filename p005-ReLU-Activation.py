"""
Why We Use Activation Functions in Neural Networks

Activation functions are mathematical functions that determine whether a neuron should be activated or not.
They are crucial components of neural networks for several key reasons:

1. **Introduce Non-linearity**: Without activation functions, neural networks would just be 
   linear transformations, no matter how many layers we stack. Linear combinations of 
   linear functions are still linear, so we'd only be able to solve linearly separable problems.

2. **Enable Complex Pattern Learning**: Non-linear activation functions allow neural networks 
   to learn and represent complex, non-linear relationships in data.

3. **Gradient Flow Control**: They help control the flow of gradients during backpropagation,
   which is essential for training deep networks.

4. **Output Range Control**: They can bound the output to specific ranges (e.g., 0-1 for probabilities).

Common Activation Functions:
- ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Sigmoid: f(x) = 1 / (1 + e^(-x))
- Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Softmax: Used for multi-class classification
"""

import numpy as np 
import nnfs # NNFS is a library for neural networks from scratch: https://github.com/Sentdex/NNfSiX
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()

X, y = spiral_data(100, 3)   


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)

#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)