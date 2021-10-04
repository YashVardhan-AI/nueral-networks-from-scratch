class Model() :
      
    def __init__(self, learning_rate, iterations) :
          
        self.learning_rate = learning_rate
        if learning_rate < 0 or learning_rate > 0.01 :
            raise ValueError("learning rate must be possitive and less then or equal to 0.01 else things die")
          
        self.iterations = iterations

    def train( self, X, Y ) :
        self.m= len(X)
        self.W = 0.0
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations) :
            self.update_weights()       
        return self

    def update_weights(self) :
             
        Y_pred = self.predict(self.X)
        cost_list = []
        for j, k in zip(Y_pred, self.Y):
            cost_list.append((k-j))
            
        avg_cost = sum(cost_list)/len(cost_list)
        r = []
        
        for o, p in zip(self.X, cost_list):
            q = o*p
            r.append(q)    
            
        dW = -(sum(r)/len(r))
        db = -avg_cost 

        self.W -= self.learning_rate * dW      
        self.b -=self.learning_rate * db

        return self
      
    def predict(self, X) :
        y = []
        for i in X:
            y_pre = i*self.W + self.b
            y.append(y_pre)
        return y
    
    def evaluate(self, X, Y) :
        Y_pred = self.predict(X)
        cost_list = []
        for t, u in zip(Y_pred, Y):
            cost_list.append((u-t))
        avg_cost = sum(cost_list)/len(cost_list)
        print("b:", self.b, "W:", self.W, "cost:", avg_cost)
    