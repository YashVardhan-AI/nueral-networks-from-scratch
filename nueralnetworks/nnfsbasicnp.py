import numpy as np

class NeuralNetwork:
    def __init__(self, x, y, learning_rate=0.1):
        self.input      = x
        self.weights1   = np.random.randn(self.input.shape[1], 6)
        self.weights2   = np.random.randn(6,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1.0/(1+ np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
        
    def feedforward(self):
        self.a1 = self.sigmoid(self.input.dot(self.weights1))
        self.output = self.sigmoid(self.a1.dot(self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.a1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.a1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1*self.learning_rate
        self.weights2 += d_weights2*self.learning_rate
        
    def fit(self, epochs):
        for i in range(epochs):
            self.feedforward()
            self.backprop()


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1],
                  [0,0,0]])
    y = np.array([[0],[1],[1],[0],[0]])
    nn = NeuralNetwork(X,y)
    nn.fit(10000)
    print(nn.output)
    