import numpy as np
#I remade the code cause I wanted to understand it better :D

class McCullochPitts:
    def __init__(self, gate, epoch):
        self.gate = gate
        self.epoch = epoch
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
    
    def train(self):
        self.set_weights()
        self.set_bias()
        self.set_data()
        for i in range(self.epoch):
            
            for j in range(len(self.data)):
                self.output = self.activation(np.dot(self.weights, self.data[j]) + self.bias)
                self.error = self.target[j] - self.output
                self.weights += self.error * self.data[j]
                self.bias += self.error
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)
        print("Output: ", self.output)
        print("Error: ", self.error)
        print("Target: ", self.target)
                
prueba = McCullochPitts("AND", 100)
prueba.train()