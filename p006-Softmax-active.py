import numpy as np
import math


#comment out parts are without numpy to apply softmax manually
# E=math.e
layer_output = [4.8, 1.21, 2.385]

# exp_values=[]
exp_values=np.exp(layer_output)

# for i in layer_output:
#     exp_values.append(E**i)

print("Exponential values:", exp_values)

# norm_base = sum(exp_values)
# norm_values = []
# for i in exp_values:
#     norm_values.append(i/norm_base)
norm_values = exp_values / np.sum(exp_values)

print("Normalized values:", norm_values)
print("Sum of normalized values:", sum(norm_values)) 