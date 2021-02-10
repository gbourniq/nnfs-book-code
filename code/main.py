"""Main file defining all neural network objects reviewed in this course"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
    """Class to build Dense Layer objects"""
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        """Initialise weights and biases"""
        self.weights: ndarray = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases: ndarray = np.zeros((1, n_neurons))
        self.output: ndarray = None

    def forward(self, inputs: ndarray) -> None:
        """Calculate layer output values from inputs, weights and biases"""
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    """Class to build the Rectified Linear Activation Function object"""
    def __init__(self) -> None:
        """Initialise output"""
        self.output: ndarray = None

    def forward(self, inputs: ndarray) -> None:
        """Calculate output values from inputs, not allowing for negative values"""
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    """
    Class to build the Softmax activation function object.
    This activation function is used downstreams for classification problem,
    where it produces some calibrated, normalised output values, ie. probabilites.
    """
    def __init__(self) -> None:
        """Initialise output"""
        self.output: ndarray = None

    def forward(self, inputs: ndarray) -> None:
        """Forward pass, which first get unormalise probabilities, by exponentiating
        the inputs, and then nromalise the values for each sample"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class BaseLoss(ABC):
    """Base class to be inherit from Loss function classes"""
    def calculate(self, y_pred: ndarray, y_true: ndarray) -> int:
        """Calculates the data and regularization losses given model
        output and ground truth values. This first calculates the
        sample losses and then the mean loss"""
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses)

    @abstractmethod
    def forward(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """Forward pass"""


class CategoricalCrossEntropyLoss(BaseLoss):
    """Class to calculate the cross entropy loss of a classification model"""
    def forward(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """Forward pass"""
        clip_value = 1e-7
        samples_count = len(y_pred)

        # clip values to prevent division by 0, both sides to not drag the mean
        y_pred = np.clip(y_pred, clip_value, 1 - clip_value)

        # Get the predicted probabilities for the actual target (true) values
        if len(y_true.shape) == 1:
            # categorical labels
            correct_confidences = y_pred[range(samples_count), y_true]
        elif len(y_true.shape) == 2:
            # one-hot encoded labels
            correct_confidences = np.sum(y_pred*y_true, axis=1)
        else:
            raise Exception(f"Invalid shape {y_true.shape}")

        # Calculate and return loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create neural network elements
# Create dense layer with 2 input features and 3 output values
dense1 = LayerDense(2,3)
# Create ReLU activation
activation1 = ActivationReLU()
# Create a second dense layer with 3 input features and 3 output values
dense2 = LayerDense(3, 3)
# Create Softmax activation
activation2 = ActivationSoftmax()
# Create loss function
loss_function = CategoricalCrossEntropyLoss()

# Forward pass of our training data
# 2 inputs features -> Dense Layer -> ReLU -> Dense Layer -> Softmax = probabilities
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
# Output of the first few samples
print(activation2.output[:5])
# Print the loss
loss = loss_function.calculate(activation2.output, y)
print(loss)
