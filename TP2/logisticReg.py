import numpy as np
import copy as cp
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iters = n_iter
        self.weights = None
        self.bias = None
        self.loss = 1e+100

    def fit(self,X,y):
        n_samples, n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for i in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)
            self.loss = self.cross_entropy(y,y_predicted)
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        

    def predict(self,X, verbose=False):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        if verbose:
            print(f"sigmoid_output : {y_predicted.shape}")
        return np.array(y_predicted_cls)
    
    
    def cross_entropy(self,y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

