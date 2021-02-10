'''
Associated YT tutorial: https://youtu.be/TEWy9vZcxW4
'''

import numpy as np 

np.random.seed(0)

# 3 samples of 4 inputs values
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        # random small initial weights in the range of [-0.10, +0.10],
        # eg. for a layer with 3 neurons
        # array([[-0.09991752, -0.08219435,  0.11352334],
        #        [-0.13830036, -0.19781719, -0.0598454 ],
        #        [-0.10535868, -0.0047565 , -0.05507077],
        #        [ 0.03366028,  0.06061087, -0.0295826 ]])
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Set biases to 0 for each of the 3 neurons: array([[0., 0., 0.]])
        # Though it can best to set to non-zero values to prevent the
        # "dead neurons" issue
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biaises
        self.output = np.dot(inputs, self.weights) + self.biases

# Example: 4 inputs -> 5 neurons layer -> 2 neurons layer = 2 outputs
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
