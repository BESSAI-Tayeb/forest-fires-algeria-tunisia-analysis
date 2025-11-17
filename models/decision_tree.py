import numpy as np
from collections import Counter


class DecisionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _gini_impurity(self, y):
        
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _split_data(self, X, y, feature_idx, threshold):
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
    
    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self._split_data(
                    X, y, feature_idx, threshold
                )
                
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                n_left, n_right = len(left_y), len(right_y)
                gini_left = self._gini_impurity(left_y)
                gini_right = self._gini_impurity(right_y)
                
                weighted_gini = (n_left / n_samples) * gini_left + \
                               (n_right / n_samples) * gini_right
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth if self.max_depth is not None else False) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = Counter(y.astype(int)).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = Counter(y.astype(int)).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        left_X, left_y, right_X, right_y = self._split_data(
            X, y, best_feature, best_threshold
        )
        
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict(self, X):
        
        X = np.array(X)
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def score(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_depth(self, node=None):
        
        if node is None:
            node = self.tree
        
        if node['leaf']:
            return 1
        
        left_depth = self.get_depth(node['left'])
        right_depth = self.get_depth(node['right'])
        
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self, node=None):
        
        if node is None:
            node = self.tree
        
        if node['leaf']:
            return 1
        
        return self.get_n_leaves(node['left']) + self.get_n_leaves(node['right'])
