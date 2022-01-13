import numpy as np



class Pocket:
    def __init__(self, eps=0.01, max_iters=100):
        self.eps = eps
        self.best_params = None
        self.loss = None
        self.n_iters = None
        self.activation_func = self._step_func
        self.max_iters = max_iters
        self.weights = None
        self.bias= None

    
    def _step_func(self, X):
        return np.where(X >= 0, 1, -1)
    
    def compute_loss(self, X, y, w, b):
        n = len(X)
        S = 0

        for i in range(n):

            linear_output = np.dot(X[i], w) + b
            y_predicted =self.activation_func(linear_output)
            if y_predicted != y[i]:
                S += 1

        self.loss = S / n

        return self
    
    
    def fit(self, X, y):
        n_samples,n_features = X.shape
        
        # Init weight and bias
        # self.weights = np.zeros(n_features)
        self._initialise_weights(X)
        self.bias = 0
        
        # w0=np.zeros(n_features)
        # b0=0

        self.compute_loss(X, y, self.weights,self.bias)
        n_iters = 0
        losses = [self.loss]

        for t in range(self.max_iters):
            
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                if self.activation_func(linear_output) * y[i] < 0:
                    self.weights = self.weights + y[i] * X[i]
                    self.bias = self.bias + y[i]

            loss = self.compute_loss(X, y, self.weights, self.bias )
            loss1 = self.compute_loss(X, y, w0, b0)
            if loss < loss1:
                w0 = self.weights
                b0 = self.bias
                loss1 = loss


            losses.append(loss1)

            n_iters = n_iters + 1
            if n_iters >= self.max_iters:
                break

        self.loss = loss1

        self.n_iters = n_iters
        print("le nombre d'iterations est: ",n_iters)

        return losses
    
    def predict(self, X):
        linear_output= np.dot(X, self.weights)+self.bias
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
        return self


