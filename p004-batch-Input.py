import numpy as np 

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [3.0, 1.0, 2.5, -1.5]]  # Batch of 4 samples with 4 features each
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output = np.dot(inputs,np.array(weights).T) + biases
print(output)