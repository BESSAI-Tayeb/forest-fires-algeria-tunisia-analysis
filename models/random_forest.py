import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_n_features(self, n_total_features):
        
        if self.max_features is None:
            return n_total_features
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_total_features)
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def _bootstrap_sample(self, X, y):
        
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _select_features(self, X):
        
        n_total_features = X.shape[1]
        n_features = self._get_n_features(n_total_features)
        
        feature_indices = np.random.choice(
            n_total_features, size=n_features, replace=False
        )
        
        return X[:, feature_indices], feature_indices
    
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            X_subset, feature_indices = self._select_features(X_sample)
            self.feature_indices.append(feature_indices)
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        
        X = np.array(X)
        
        tree_predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_indices]
            tree_predictions.append(tree.predict(X_subset))
        
        tree_predictions = np.array(tree_predictions)
        
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        
        X = np.array(X)
        
        tree_predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_indices]
            tree_predictions.append(tree.predict(X_subset))
        
        tree_predictions = np.array(tree_predictions)
        
        probabilities = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            prob_class_1 = np.mean(votes == 1)
            prob_class_0 = 1 - prob_class_1
            probabilities.append([prob_class_0, prob_class_1])
        
        return np.array(probabilities)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def feature_importance(self, X, y):
        
        n_features = X.shape[1]
        feature_counts = np.zeros(n_features)
        
        for feature_indices in self.feature_indices:
            for idx in feature_indices:
                feature_counts[idx] += 1
        
        if feature_counts.sum() > 0:
            importances = feature_counts / feature_counts.sum()
        else:
            importances = np.zeros(n_features)
        
        return importances
