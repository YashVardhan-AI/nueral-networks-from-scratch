from jax import numpy as np
import jax
from jax import random
from functools import partial
class nn():
    def __init__(self, X, y, lr = 0.01, epochs = 10000, dims = 2, layers = 2):
        self.X = X
        self.y = y
        self.lr = lr
        for self.i in range(layers):
            self.i = random.normal(random.PRNGKey(0), (X.shape[1], dims))
            print(self.i)
        self.epochs = epochs    
    
    
    @partial(jax.jit, static_argnums=(0,))
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))


    def feedforward(self):
        self.l1 = self.sigmoid(np.dot(self.X, self.w1))
        self.output = self.sigmoid(np.dot(self.l1, self.w2))
        
    @partial(jax.jit, static_argnums=(0,))
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
        

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.l1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.X.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.w2.T) * self.sigmoid_derivative(self.l1)))

        # update the weights with the derivative (slope) of the loss function
        self.w1 += d_weights1*self.lr
        self.w2 += d_weights2*self.lr
                
    def fit(self):
        for i in range(self.epochs):
            self.feedforward()
            self.backprop()
            
    def predict(self):
        return self.output
    
    
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1],
                  [0,0,0]])
    y = np.array([[0],[1],[1],[0],[0]])
    nn = nn(X,y)
    nn.fit()
    print(nn.output)
    
    