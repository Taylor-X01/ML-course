import numpy as np


class Adaline(object):
    """ Adaline (Adaptive Linear Neuron) for binary classification.
        Minimises the cost function using gradient descent. """

    def __init__(self, learn_rate = 0.1, iterations = 100):
        self.learn_rate = learn_rate
        self.iterations = iterations


    def fit(self, X, y, biased_X = False, standardised_X = False):
        """ Fit training data to our model """
        n_samples,n_features = X.shape
        
        if not standardised_X:
            X = self._standardise_features(X)
        if not biased_X:
            X = self._add_bias(X)
        self._initialise_weights(X)
        self.W = [self.weights]
        self.cost = []

        for cycle in range(self.iterations):
            output_pred = self._activation(self._net_input(X))
            errors = y - output_pred
            # if (errors == np.zeros(errors.shape)).all():
            self.weights += self.learn_rate * np.dot(errors,X)
            cost = (errors**2).sum() / n_samples
            self.cost.append(cost)
            self.W.append(self.weights)
        return self


    def _net_input(self, X):
        """ Net input function (weighted sum) """
        return np.dot(X, self.weights)

    def _thresh_func(self, X):
        return np.where(X >= 0, 1, -1)

    def predict(self, X, biased_x = False):
        if not biased_x:
            X = self._add_bias(X)
        linear_output = self._net_input(X)
        y_predicted = self._thresh_func(linear_output)

        return y_predicted

    # def predict(self, X, biased_X=False):
    #     """ Make predictions for the given data, X, using unit step function """
    #     if not biased_X:
    #         X = self._add_bias(X)
    #     return np.where(self._activation(self._net_input(X)) >= 0.0, 1, -1)


    def _add_bias(self, X):
        """ Add a bias column of 1's to our data, X """
        bias = np.ones((X.shape[0], 1))
        biased_X = np.hstack((bias, X))
        return biased_X


    def _initialise_weights(self, X):
        """ Initialise weigths - normal distribution sample with standard dev 0.01 """
        random_gen = np.random.RandomState(1)
        self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        return self
    
    
    def _standardise_features(self, X):
        """ Standardise our input features with zero mean and standard dev of 1 """
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis = 0)
        return X_norm


    def _activation(self, X):
        """ Linear activation function - simply returns X """
        return X

      
