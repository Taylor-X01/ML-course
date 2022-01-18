import numpy as np
import copy as cp


class Pocket:
    def __init__(self, eps=0.01, T_max=100):
        self.eps = eps
        self.best_params = None
        self.loss = None
        self.n_iters = None
        self.activation_func = self._step_func
        self.max_iters = T_max
        self.weights = None
        self.AllL = None
        # self.bias= None

    
    def _step_func(self, X):
        return np.where(X >= 0, 1, -1)
    
    def compute_loss(self, X, y, w_tmp, b_tmp):
        n = len(X)
        S = 0
        
        w = w_tmp
        b = b_tmp

        for i in range(n):

            linear_output = np.dot(X[i], w) + b
            y_predicted =self.activation_func(linear_output)
            if y_predicted != y[i]:
                S += 1

        loss = S / n

        return loss
    
    
    def fit(self, X, y):
        n_samples,n_features = X.shape
        
        # Init weight and bias
        self._initialise_weights(X)

        self.loss = self.compute_loss(X, y,self.weights[1:],self.weights[0]) # Ls_0
        n_iters = 0
        losses = [cp.deepcopy(self.loss)]
        Weights = [cp.deepcopy(self.weights)]
        self.best_params = [cp.deepcopy(self.weights)]
        self.AllL = [cp.deepcopy(self.loss)]
        for t in range(self.max_iters):
            # get a new weight and bias
            self._pla_training_cycle(X,y)
            # Test them
            loss = self.compute_loss(X, y, self.weights[1:] ,self.weights[0])
            self.AllL.append(cp.deepcopy(loss))
            # Update the Weights and bias if the loss is good
            if loss < self.loss:
                self.loss = loss
                self.best_params.append(cp.deepcopy(self.weights))
                

            losses.append(cp.deepcopy(self.loss))
            Weights.append(cp.deepcopy(self.weights))

            # n_iters = n_iters + 1
            # if n_iters >= self.max_iters:
            #     break

     

        # self.n_iters = n_iters
        # print("le nombre d'iterations est: ",n_iters)

        return [np.array(Weights),losses]
    
    
    
    def _pla_training_cycle(self,X,y):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        for i in range(X.shape[0]):
            if self.activation_func(linear_output[i]) * y[i] < 0:
                self.weights[1:] = self.weights[1:] + y[i] * X[i]
                self.weights[0] = self.weights[0] + y[i]
        return self

        
 
    def predict(self, X):
        linear_output= np.dot(X, self.weights[1:])+self.weights[0]
        y_predicted= self.activation_func(linear_output)
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
        # self.weights = np.zeros(1 + X.shape[1])
        return self


