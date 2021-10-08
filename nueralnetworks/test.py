from jax import numpy as np
import jax
from jax import random
from functools import partial

class nn():
    def __init__(self, X, y, lr = 0.01, epochs = 10000, dims = 2, layers = 2):
        self.X = X
        self.y = y
        self.lr = lr
        self.hiddenlayer = []
        for self.i in range(layers):
            w = random.normal(random.PRNGKey(0), (X.shape[1], dims))
            self.layervals.append(w)
        self.epochs = epochs 
        print(self.layervals)

X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1],
                  [0,0,0]])
y = np.array([[0],[1],[1],[0],[0]])


nueralnetworks = nn(X, y, lr = 0.01, epochs = 10000, dims = 2, layers = 2)
