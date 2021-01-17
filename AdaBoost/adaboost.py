import numpy as np

from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, base_estimator, n_estimators=10, **params):
        """
        n_estimators : int
            number of base estimators
        base_estimator : class
            class for base_estimator with fit(), predict() and predict_proba() methods
        n_estimators : int
            number of base_estimators
        params : kwargs
            params for base_estimator initialization
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimators = []
        self.alphas = []
        self.weights = None
        self.params = params

    def _error(self, y_true, y_pred):
        return self.weights.dot(y_true != y_pred)

    def fit(self, X, y):
        """
            y = {+1, -1}
        """
        self.estimators = []
        self.alphas = []
        self.weights = np.array([1/X.shape[0]] * X.shape[0])

        for _ in range(self.n_estimators):
            estimator = self.base_estimator(**self.params)
            estimator.fit(X, y, sample_weight=self.weights)
            predict = estimator.predict(X)
        
            error = self._error(y, predict)
            alpha = 1/2 * np.log((1-error)/error)

            self.weights = self.weights * np.exp(-alpha * y * predict)
            self.weights = self.weights / self.weights.sum()

            self.estimators.append(estimator)
            self.alphas.append(alpha)
    
        return self

    def predict_proba(self, X):
        if not (0 < len(self.estimators) == len(self.alphas)):
            raise RuntimeError('Bagger is not fitted', (len(self.estimators), len(self.alphas)))
          
        predict = np.zeros(X.shape[0])  
        for estimator, alpha in zip(self.estimators, self.alphas):
            predict = predict + alpha * estimator.predict(X)
        return 1/(1 + np.exp(-predict))
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


class AdaBoostClassifier(AdaBoost):
    def __init__(self, n_estimators=30, max_depth=None, min_samples_leaf=1, random_state=None, **params):
        base_estimator = DecisionTreeClassifier

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **params,
        )
