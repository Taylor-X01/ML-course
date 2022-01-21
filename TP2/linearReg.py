import numpy as np
import copy as cp
import random
random.seed(0)

class linear_regression:

    def __init__(self,lr=0.001, n_iters=2000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.compute_loss = self.MSE
        self.weight = None
        self.loss = None
    
    def fit(self, X, Y, verbose = False):
        """ Fit training data to our model """
        n_samples,n_features = X.shape
        
        # Init the weights and bias
        self._initialise_weights(X)
        
        # Compute the loss function
        self.compute_loss(X, Y)
        
        Weights = [cp.deepcopy(self.weight)]
        Losses = [cp.deepcopy(self.loss)]
        
        
        for _ in range(self.n_iters):
            
            if verbose:
                print(f"Weight : {self.weight}")
            Y_hat = np.dot(X,self.weight[1:]) + self.weight[0]
            
            dw = (2/n_features) * np.dot(X.T,(Y_hat - Y))
            db = (2/n_features) * np.sum(Y_hat - Y,axis=0)
            if verbose:
                print(f"\t dw : {dw}\n\t db : {db}")
            
            # Updating
            self.weight[1:] += -self.lr * dw
            self.weight[0] += -self.lr * db

            self.compute_loss(X, Y)
            Losses.append(cp.deepcopy(self.loss))
            Weights.append(cp.deepcopy(self.weight))
        return [Weights,Losses]

    
    def MSE(self,X, Y):
        Y_hat = np.dot(X,self.weight[1:]) + self.weight[0]
        self.loss = np.mean((Y-Y_hat)**2)*(1/Y.shape[0])  
        return self

        # diff = Y - Y_hat
        # diff_sqrd = diff**2
        # return (1/Y.shape[0])*np.sum(diff_sqrd)


    def predict(self, X):
        y_hat = np.dot(X,self.weight[1:]) + self.weight[0]
        return y_hat

    
        
    def _initialise_weights(self,X):
        """create vector of random weights
        Parameters
        ----------
        X: 2-dimensional array, shape = [n_samples, n_features]
        Returns
        -------
        w: array, shape = [w_bias + n_features]"""
        rand = np.random.RandomState(1)
        self.weight = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        return self
    
