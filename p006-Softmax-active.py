import numpy as np
import math


#comment out parts are without numpy to apply softmax manually
# E=math.e
# layer_output = [4.8, 1.21, 2.385]
layer_output =[[4.8, 1.21, 2.385],
               [8.9, -1.81, 0.2],
               [1.41,1.051,0.026]]  # Example output from a layer
# exp_values=[]
exp_values=np.exp(layer_output)

# for i in layer_output:
#     exp_values.append(E**i)

print("Exponential values:", exp_values)

# norm_base = sum(exp_values)
# norm_values = []
# for i in exp_values:
#     norm_values.append(i/norm_base)
# norm_values = exp_values / np.sum(exp_values) #this is for single layer output in numpy method
norm_values =exp_values / np.sum(exp_values, axis=1, keepdims=True) #this is for multiple layer output

print("Normalized values:", norm_values)
print("Sum of normalized values:", sum(norm_values)) 