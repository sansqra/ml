import numpy as np

X = [
        [1.3, 2.0, 3.2, 5],
        [2.0, 5.0, 1.0, 1.5],
        [-1.5, 2.7, 3.3, -0.8]
    ]

class Dense_Layer:

    def __init__(self, inputs, neurons):
        self.weights = 0.10 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)

l1 = Dense_Layer(4, 5)
l2 = Dense_Layer(5, 2)

l1.forward_pass(X)
print(l1.output)