import numpy as np


class Perceptron:
    def __init__(self, gate, epoch, learning_rate):
        self.gate = gate
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.bias = 0
        self.weights = []
        self.data = []
        self.target = []
        self.output = []
        self.error = []

    def set_weights(self):
        if self.gate == "NOT":
            self.weights = np.random.uniform(-1, 1, 1)
        else:
            self.weights = np.random.uniform(-1, 1, 2)

    def set_bias(self):
        self.bias = 0

    def set_data(self):
        if self.gate == "AND":
            self.data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            self.target = np.array([0, 0, 0, 1])
        elif self.gate == "OR":
            self.data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            self.target = np.array([0, 1, 1, 1])
        elif self.gate == "NOT":
            self.data = np.array([[0], [1]])
            self.target = np.array([1, 0])

    def activation(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def perceptronRule(self, j):
        if self.target[j] == 1 and self.activation(np.dot(self.weights, self.data[j]) + self.bias) == 0:
            self.weights += self.learning_rate * self.data[j]
            self.bias += self.learning_rate
        elif self.target[j] == 0 and self.activation(np.dot(self.weights, self.data[j]) + self.bias) == 1:
            self.weights -= self.learning_rate * self.data[j]
            self.bias -= self.learning_rate
        else:
            pass

    def train(self):
        self.set_weights()
        self.set_bias()
        self.set_data()
        for i in range(self.epoch):
            for j in range(len(self.data)):
                self.output = self.activation(
                    np.dot(self.weights, self.data[j]) + self.bias)
                self.error = self.target[j] - self.output
                self.perceptronRule(j)
            print("--------------------------------------- ")
            print("Epoch: ", i)
            print("---------------------------------------")
            print("Weights: ", self.weights)
            print("Bias: ", self.bias)
            print("     Output: ", self.output)
            print("     Target: ", self.target[j])
            print("         Error: ", self.error)
            




prueba = Perceptron("AND", 10, 0.1)
prueba.train()
