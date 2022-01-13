import numpy as np
import copy as cp
import random
random.seed(0)

class Adaline:
    """ Adaline (Adaptive Linear Neuron) for binary classification.
        Minimises the cost function using gradient descent. """
    
    def __init__(self, eps=0.01, max_iters=100):
        self.eps = eps
        self.loss = None
        self.n_iters = None
        self.activation_func=self._unit_func
        self.thresh_func = self._step_func
        self.max_iters = max_iters
        self.weights = None

    
    def _step_func(self, X):
        return np.where(X >= 0, 1, -1)
    
    def _unit_func(self,X):
        return X
    
    def compute_loss(self, X, y): # (MSE) J = 1/n * sum( (y_i - y_i_hat)**2 )
        n = len(y)
        w = self.weights

        net_input = np.dot(X, w[1:]) + w[0]
        S = ((y -  net_input)**2).sum()
        self.loss = S / n

        return self
    
    
    def fit(self, X, y):
        """ Fit training data to our model """
        n_samples,n_features = X.shape
        
        # Init the weights and bias
        self._initialise_weights(X)
        
        # Compute the loss function
        self.compute_loss(X, y)
        n_iters = 0
        
        # Init losses & Weights list
        w_i = cp.deepcopy(self.weights)
        Weights = [w_i]
        losses = [self.loss]
        for t in range(self.max_iters):# stop condition
            
          # Training cycle function (Matricial compute)
            self._adaline_training_cycle(X,y)
            
          # Training cycle (Itirative compute)
            # for i in range(n_samples):
            #     # Compute Net Input
            #     net_input = np.dot(X[i], self.weights)
            #     # Compute the linear output y_hat
            #     linear_output =self.activation_func( net_input )
            #     # Error = y_true - y_hat
            #     e_i = (y[i] - linear_output)**2
            #     # Weight and bias update
            #     # if e_i != 0:
            #     self.weights -= self.eps * 2 * e_i * X[i]
  
            w_i = cp.deepcopy(self.weights)
            Weights.append(w_i)
        
            self.compute_loss(X, y)
            losses.append(self.loss)

        print("Total Number of Epochs:", self.max_iters)
        return [Weights,losses]
    

    def _adaline_training_cycle(self, X, y, verbose=False):
        n_samples = X.shape[0]
        # Compute Net Input
        net_input = np.dot(X, self.weights[1:]) + self.weights[0]
        # Compute the linear output y_hat 
        linear_output =self.activation_func( net_input ) ## f(x) = x
        # Compute the gradient
        gradient = - 2*( np.dot((y - linear_output), X))
        
        self.weights[1:] += self.eps * (-gradient)
        self.weights[0] += self.eps * (y - linear_output).sum()
        
        if verbose:
            print(f"Gradient :{gradient}")
            print(f"Weights : {self.weights}")
        return self
        
    
    def predict(self, X):
        
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        y_predicted = self.thresh_func(linear_output)

        return y_predicted
    
    
    

    
    def _initialise_weights(self,X):
        """create vector of random weights
        Parameters
        ----------
        X: 2-dimensional array, shape = [n_samples, n_features]
        Returns
        -------
        w: array, shape = [w_bias + n_features]"""
        rand = np.random.RandomState(1)
        self.weights = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        return self
    




