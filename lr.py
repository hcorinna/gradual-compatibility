import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import it
import config

class CustomLogisticRegression:
    """
    Based on https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/.
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization

    Regularization parameters:
    alpha -- regularizes independence term (default 0)
    beta -- regularizes balance term (default 0)
    gamma -- regularizes negative accuracy term (default 0)
    """
    def __init__(self,
                 X=None, Y=None, A=None, weights_init=None,
                 alpha=0, beta=0, gamma=0, _lambda=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._lambda = _lambda
        self.weights = None
        self.weights_init = weights_init
        
        self.X = X
        self.Y = Y
        self.A = A
    
    def sigmoid(self, weights):
        return 1 / (1 + np.exp(-weights))
    
    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.weights))

    def custom_loss(self, weights):
        self.weights = weights
        y_prob = self.predict_prob(self.X)
        
        # probabilities to binary predictions
        y_pred = y_prob >= 0.5
        y_pred = y_pred.astype(int)
        
        loss = log_loss(self.Y, y_prob)
        if self.alpha != 0:
            loss += self.alpha * it.independence(list(self.Y), y_pred, self.A)
        if self.beta != 0:
            loss += self.beta * it.balance(list(self.Y), y_pred, self.A)
        if self.gamma != 0:
            loss += self.gamma * it.neg_accuracy(list(self.Y), y_pred)
        if self._lambda != 0:
            loss += self._lambda * self.l2(self.weights)
        return loss
    
    def l2(self, weights):
        return sum(np.array(weights)**2)
    
    def fit(self, maxiter=1000):        
        # Initialize weight estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.weights_init)==type(None):
            # set weights_init = 1 for every feature
            self.weights_init = np.array([1]*self.X.shape[1])
        else: 
            # Use provided initial values
            pass
            
        if self.weights!=None and all(self.weights_init == self.weights):
            print("Model already fit once; continuing fit with more itrations.")
            
        res = minimize(self.custom_loss, self.weights_init,
                       method=config.method, options={'maxiter': maxiter})
        self.weights = res.x
        self.weights_init = self.weights