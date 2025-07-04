import numpy as np 

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
        #shape of the weights matrix is (n_inputs, n_neurons)
        #n_inputs is the number of inputs to the layer, n_neurons is the number of neurons in the layer
        #weights are initialized with small random values
        self.biases = np.zeros((1, n_neurons))
        #shape of the biases matrix is (1, n_neurons)
        #biases are initialized to zero
        # why zeros? because we want to learn the biases from the data, starting from zero
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)   # 4 inputs, 5 neurons
layer2 = Layer_Dense(5,2)   # 5 inputs (from layer1), 2 neurons 

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)