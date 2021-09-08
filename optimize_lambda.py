import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import lr
import config

class CustomCrossValidator:
    
    """
    Based on https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/.
    Cross validates arbitrary model using MAPE criterion on
    list of lambdas.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def cross_validate(self, lambdas, num_folds=config.number_of_lambda_folds):
        """
        lambdas: set of regularization parameters to try
        num_folds: number of folds to cross-validate against
        """
        
        self.lambdas = lambdas
        self.cv_scores = []
        X = self.X.to_numpy()
        Y = self.Y.to_numpy()
        
        for lam in self.lambdas:
            print("Lambda: {}".format(lam))
            
            weights_init = None
            
            # Split data into training/holdout sets
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
            kf.get_n_splits(X)
            
            # Keep track of the error for each holdout fold
            k_fold_scores = []
            
            # Iterate over folds, using k-1 folds for training
            # and the k-th fold for validation
            f = 1
            for train_index, test_index in kf.split(X):
                # Training data
                CV_X = X[train_index,:]
                CV_Y = Y[train_index]
                
                # Holdout data
                holdout_X = X[test_index,:]
                holdout_Y = Y[test_index]
                
                # Fit model to training sample
                lambda_fold_model = lr.CustomLogisticRegression(
                    X=CV_X,
                    Y=CV_Y,
                    A=None,
                    weights_init=weights_init,
                    alpha=0,
                    beta=0,
                    gamma=0,
                    _lambda=lam
                )
                lambda_fold_model.fit()
                
                # Calculate holdout error
                fold_probs = lambda_fold_model.predict_prob(holdout_X)
                fold_cross_entropy = log_loss(holdout_Y, fold_probs)
                k_fold_scores.append(fold_cross_entropy)
                print("Fold: {}. Error: {}".format(f, fold_cross_entropy))
                f += 1
            
            # Error associated with each lambda is the average
            # of the errors across the k folds
            lambda_scores = np.mean(k_fold_scores)
            print("LAMBDA AVERAGE: {}".format(lambda_scores))
            self.cv_scores.append(lambda_scores)
        
        # Optimal lambda is that which minimizes the cross-validation error
        self.lambda_star_index = np.argmin(self.cv_scores)
        self.lambda_star = self.lambdas[self.lambda_star_index]
        print("\n\n**OPTIMAL LAMBDA: {}**".format(self.lambda_star))
        return self.lambda_star