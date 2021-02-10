import numpy as np 
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

"""
Basically activation functions are required to solve non-linear problems, some examples are
- Step function (y=1 when x>0, y=0 when x<=0)
- Linear (x=y)
- Sigmoid (basically a "more information" kind of step function), where y -> 0 when x -> -inf
- Rectified linear (Relu): (y=x when x>0, y=0 when x<=0)
- Softmax: Unlike other activation function, this one is used at the end of the network for
        classification problem, where it produces some calibrated, normalised values for pred probabilites
"""

nnfs.init()

# X = inputs: sets of (x,y) coordinates
# y = target: target Categories: 1, 2 or 3
X, y = spiral_data(100, 3)   

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        # Relu: y=x when x>0, y=0 when x<=0; basically does not allow for negative values
        self.output = np.maximum(0, inputs)

# Initialise objects
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

# Inputs -> Dense Layer -> Activation function
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
