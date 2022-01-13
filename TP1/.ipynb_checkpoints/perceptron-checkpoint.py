import numpy as np
import random
random.seed(0)


class Perceptron:
    def __init__(self, eps=0.01, max_iters=2000):
        self.eps = eps
        # self.best_params = None
        self.losses = None
        self.n_iters = None
        self.activation_func=self._step_func
        self.max_iters = max_iters
        self.weights = None
        self.bias= None

    
    def _step_func(self, X):
        return np.where(X >= 0, 1, -1)

    def compute_loss(self, X, y, w, b): # Ls = |xi ; hs(xi) != yi| / |s|
        n = len(X)
        S = 0

        for i in range(n):

            linear_output = np.dot(X[i], w) + b
            y_predicted =self.activation_func(linear_output)
            if y_predicted != y[i]:
                S += 1

        Ls = S / n

        return Ls
    

    

    def fit(self, X, y):
        
        n_samples,n_features = X.shape
        
        # Init weights and bias
        # self.weights=np.zeros(n_features)
        # self.weights=[random.random()*1 for i in range(n_features)]
        # self.weights=np.ones(n_features)
        self._initialise_weights(X)
        self.bias= 0
        
        # Compute the loss function
        loss = self.compute_loss(X, y, self.weights,self.bias)
        n_iters = 0
        Weights = [(self.weights,self.bias)]
        self.losses = [loss]
        
        
        while loss >= self.eps: # Stop condition
            
            for i in range(n_samples):
                # Compute the linear output
                linear_output = np.dot(X[i], self.weights) + self.bias
                
                # Adjusting weights and bias
                if self.activation_func(linear_output) * y[i] < 0: # If the point is misclassified
                    self.weights = self.weights + y[i] * X[i]   # w = w + yi.xi
                    self.bias = self.bias + y[i]                # b = b + yi
                    
                    
            Weights.append((self.weights,self.bias))
            

            loss = self.compute_loss(X, y, self.weights, self.bias )

            self.losses.append(loss)

            n_iters = n_iters + 1
            
            if n_iters >= self.max_iters:
                break

        self.n_iters = n_iters
        print("le nombre d'iterations est: ", self.n_iters)
        
        return [Weights,self.losses,self.n_iters]

    
    def predict(self, X):
        linear_output= np.dot(X, self.weights)+self.bias
        y_predicted= self.activation_func(linear_output)
        return y_predicted
        
    def _initialise_weights(self, X):
        """ Initialise weigths - normal distribution sample with standard dev 0.01 """
        random_gen = np.random.RandomState(1)
        self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        return self







