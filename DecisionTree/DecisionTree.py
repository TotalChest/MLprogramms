import numpy as np


class BasePredictionMatcher:
    def get_prediction_value(self, y):
        raise NotImplementedError


class ClassificationProbabilityMatcher(BasePredictionMatcher):
    def __init__(self, n_classes=None):
        self.n_classes = n_classes
    
    def get_prediction_value(self, y):
        """
        Parameters
        ----------
        y : numpy ndarray of shape (n_objects,) 
        
        Returns
        -------
        classes_probabilities : numpy ndarray of shape (n_classes,)
            classes_probabilities[i] - ith class probability
        """

        return np.bincount(y, minlength=self.n_classes) / y.size


class BaseInformationCriterion:
    def find_best_split(self, X_feature, y):
        """
        Parameters
        ----------
        X_feature : numpy ndarray of shape (n_objects,)
        
        y : numpy ndarray of shape (n_objects,)
        
        Returns
        -------
        best_threshold : float 
        best_Q_value : float
        """
        possible_thresholds = sorted(list(set(X_feature)))[1:]
        H_all_X = self.get_H(y)
        
        best_threshold = None
        best_Q_value = -float('inf')
        
        for threshold in possible_thresholds:
            curr_mask = (X_feature >= threshold)
            y_left = y[~curr_mask]
            y_right = y[curr_mask]
            Q_value = self.get_Q(H_all_X, y_left, y_right)
            if Q_value > best_Q_value:
                best_Q_value = Q_value
                best_threshold = threshold

        return best_threshold, best_Q_value
        
    def get_Q(self, H_main, y_left, y_right):
        """
        Parameters
        ----------
        H_main : float
        
        y_left : numpy ndarray of shape (n_objects_left,)
        
        y_right : numpy ndarray of shape (n_objects_right,)
        
        Returns
        -------
        Q_value : float
        """
        total_L, total_R = len(y_left), len(y_right)
        total = total_L + total_R
        
        return H_main - \
               self.get_H(y_left) * total_L / total - \
               self.get_H(y_right) * total_R / total

    def get_H(self, y):
        raise NotImplementedError


class GeanyInformationCriterion(BaseInformationCriterion):
    def __init__(self, n_classes=None):
        self.probability_matcher = ClassificationProbabilityMatcher(n_classes=n_classes)
        
    def get_H(self, y):
        """
        Parameters
        ----------
        y : numpy ndarray of shape (n_objects,)
        
        Returns
        -------
        H_value : float
        """
        p = self.probability_matcher.get_prediction_value(y)
        return p @ (1-p)


class TreeNode:
    def __init__(self, prediction_value, feature_index=None, threshold=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.prediction_value = prediction_value
        self.right_children = None
        self.left_children = None
        
    @property
    def is_terminal(self):
        if self.right_children is None and self.left_children is None:
            return True
        return False
    
    def get_next_nodes_mask(self, X):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        
        Returns
        -------
        next_node_indexes : numpy ndarray of shape (n_objects,)
        """
        feat = X[:, self.feature_index]
        ans = feat >= self.threshold
        return ans
        
    def split_node(self, X, y, criterion, prediction_matcher):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)
        criterion : instance of BaseInformationCriterion inherited
        prediction_matcher : instance of BasePredictionMatcher inherited
        
        Returns
        -------
        indexes_mask : numpy ndarray of shape (n_objects,)
            Right children objects mask.
        right_children : instance of TreeNode
        left_children : instance of TreeNode
        """
        best_feature_index = None
        best_Q = -float('inf')
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            feature_threshold, feature_Q = criterion.find_best_split(X[:, feature_index].ravel(), y)
            if feature_Q > best_Q:
                best_Q = feature_Q
                best_threshold = feature_threshold
                best_feature_index = feature_index

        self.feature_index = best_feature_index
        self.threshold = best_threshold
        
        indexes_mask = self.get_next_nodes_mask(X)
        right_pred_value = prediction_matcher.get_prediction_value(y[indexes_mask])
        left_pred_value = prediction_matcher.get_prediction_value(y[~indexes_mask])
        self.right_children = TreeNode(right_pred_value)
        self.left_children = TreeNode(left_pred_value)
        
        return indexes_mask, self.right_children, self.left_children


class DecisionTree:
    def __init__(
        self, criterion, prediction_matcher,
        max_depth, min_leaf_size,
    ):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.criterion = criterion
        self.prediction_matcher = prediction_matcher
    
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)
        """
        base_answer = self.prediction_matcher.get_prediction_value(y)
        self.root_node = TreeNode(prediction_value=base_answer)
        
        current_objects_indexes = np.arange(0, X.shape[0])
        self._build_new_nodes(X, y, current_objects_indexes, self.root_node, 0)
        
    def _build_new_nodes(self, X, y, current_objects_indexes, current_node, current_depth):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)
        current_objects_indexes : numpy ndarray of shape (n_objects_node,)
            Indexes of current node objects
        current_node : instance of TreeNode
        current_depth : int
        """
        if current_depth == self.max_depth or \
           current_objects_indexes.size <= self.min_leaf_size or \
           (current_node.prediction_value == 1.0).any():
            return
        new_X = X[current_objects_indexes, :]
        new_y = y[current_objects_indexes]
        mask, right_children, left_children = (
            current_node.split_node(new_X, new_y, self.criterion,
                                    self.prediction_matcher)
        )
        self._build_new_nodes(X, y, current_objects_indexes[mask], right_children, current_depth + 1)
        self._build_new_nodes(X, y, current_objects_indexes[~mask], left_children, current_depth + 1)
        
    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        """
        predictions_dict = dict()
        
        current_objects_indexes = np.arange(0, X.shape[0])
        self._get_predictions_from_terminal_nodes(X, current_objects_indexes,
                                                  self.root_node, predictions_dict)
        
        predictions = np.array([
            elem[1] for elem in sorted(predictions_dict.items(), key=lambda x: x[0])
        ])
        return predictions
    
    def _get_predictions_from_terminal_nodes(self, X, current_objects_indexes, current_node,
                                             current_predictions):
        """
        Parameters
        ----------
        X : numpy ndarray of shape (n_objects, n_features)
        current_objects_indexes : numpy ndarray of shape (n_objects_node,)
        current_node : instance of TreeNode
        current_predictions : dict
        """
        
        if current_node.is_terminal:
            for index in current_objects_indexes:
                current_predictions[index] = current_node.prediction_value
        else:
            mask = current_node.get_next_nodes_mask(X[current_objects_indexes])
            self._get_predictions_from_terminal_nodes(X, current_objects_indexes[mask], current_node.right_children, current_predictions)
            self._get_predictions_from_terminal_nodes(X, current_objects_indexes[~mask], current_node.left_children, current_predictions)


class ClassificationDecisionTree(DecisionTree):
    def __init__(self, max_depth, min_leaf_size):
        criterion = GeanyInformationCriterion()
        matcher = ClassificationProbabilityMatcher()
        
        super().__init__(
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            criterion=criterion,
            prediction_matcher=matcher,
        )
        
    def fit(self, X, y):
        n_classes = y.max() + 1
        self.prediction_matcher.n_classes = n_classes
        
        super().fit(X, y)
    
    def predict_proba(self, X):
        return super().predict(X)
    
    def predict(self, X):
        probabilities = super().predict(X)
        return probabilities.argmax(axis=1)


class RegressionPredictionMatcher(BasePredictionMatcher):
    def get_prediction_value(self, y):
        return y.mean()


class RegressionInformationCriterion(BaseInformationCriterion):
    def __init__(self):
        self.regression_matcher = RegressionPredictionMatcher()
        
    def get_H(self, y):
        c = self.regression_matcher.get_prediction_value(y)
        return (y - c).sum() / y.size


class RegressionDecisionTree(DecisionTree):
    def __init__(self, max_depth, min_leaf_size):
        criterion = RegressionInformationCriterion()
        matcher = RegressionPredictionMatcher()
        
        super().__init__(
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            criterion=criterion,
            prediction_matcher=matcher,
        )
        
    def fit(self, X, y):
        super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
