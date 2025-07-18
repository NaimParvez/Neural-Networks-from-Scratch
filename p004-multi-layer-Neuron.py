import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [3.0, 1.0, 2.5, -1.5]]


weights1 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -1.0, 2.0, -0.91],
    [0.26, -0.27, 0.17, 0.87]
]
biases1 = [2.0, 3.0, 0.5]


weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0,44,0.73,-0.13]
]
biases2 = [-1.0, 2.0,-0.5]

# Forward pass through layer 1
layer1_outputs = np.dot(inputs,np.array(weights1).T) + biases1

# Forward pass through layer 2
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T) + biases2

print(layer2_outputs)
