# Custom Machine Learning Models Documentation

This document describes the custom implementations of three machine learning models built from scratch without using scikit-learn.

---

## Table of Contents

1. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
2. [Decision Tree](#decision-tree)
3. [Random Forest](#random-forest)
4. [Usage Examples](#usage-examples)

---

## K-Nearest Neighbors (KNN)

### Overview
A non-parametric classification algorithm that predicts the class of a sample based on the majority class among its k nearest neighbors in the feature space.

### Class: `KNN`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 5 | Number of nearest neighbors to consider for voting |
| `distance_metric` | str | 'euclidean' | Distance metric to use ('euclidean' or 'manhattan') |

#### Methods

**`fit(X, y)`**
- **Purpose**: Store the training data (KNN is a lazy learner)
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Training feature matrix
  - `y`: numpy.ndarray of shape `(n_samples,)` - Training target labels (0 or 1)
- **Output**: Returns `self`

**`predict(X)`**
- **Purpose**: Predict class labels for test samples
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
- **Output**: numpy.ndarray of shape `(n_samples,)` - Predicted class labels (0 or 1)

**`predict_proba(X)`**
- **Purpose**: Predict class probabilities
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
- **Output**: numpy.ndarray of shape `(n_samples, 2)` - Probabilities `[P(class=0), P(class=1)]`

**`score(X, y)`**
- **Purpose**: Calculate accuracy score
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
  - `y`: numpy.ndarray of shape `(n_samples,)` - True labels
- **Output**: float - Accuracy (fraction of correct predictions)

### Algorithm Details

1. **Distance Calculation**: Computes distance between test sample and all training samples
2. **Neighbor Selection**: Selects k samples with smallest distances
3. **Voting**: Predicts class based on majority vote among k neighbors
4. **Probability**: Calculates probability as proportion of each class among k neighbors

### When to Use

- **Pros**:
  - Simple and intuitive
  - No training phase
  - Works well with small datasets
  - Non-parametric (makes no assumptions about data distribution)

- **Cons**:
  - Computationally expensive for large datasets (O(n) prediction time)
  - Sensitive to feature scaling
  - Performance degrades in high dimensions (curse of dimensionality)
  - Memory intensive (stores all training data)

---

## Decision Tree

### Overview
A tree-based model that recursively splits the feature space to create decision rules for classification using the CART (Classification and Regression Trees) algorithm with Gini impurity as the splitting criterion.

### Class: `DecisionTree`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int or None | None | Maximum depth of the tree. If None, nodes expand until pure or min_samples_split reached |
| `min_samples_split` | int | 2 | Minimum number of samples required to split an internal node |
| `min_samples_leaf` | int | 1 | Minimum number of samples required to be at a leaf node |

#### Methods

**`fit(X, y)`**
- **Purpose**: Build the decision tree from training data
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Training feature matrix
  - `y`: numpy.ndarray of shape `(n_samples,)` - Training target labels (0 or 1)
- **Output**: Returns `self`

**`predict(X)`**
- **Purpose**: Predict class labels by traversing the tree
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
- **Output**: numpy.ndarray of shape `(n_samples,)` - Predicted class labels (0 or 1)

**`score(X, y)`**
- **Purpose**: Calculate accuracy score
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
  - `y`: numpy.ndarray of shape `(n_samples,)` - True labels
- **Output**: float - Accuracy (fraction of correct predictions)

**`get_depth()`**
- **Purpose**: Get the maximum depth of the tree
- **Input**: None
- **Output**: int - Tree depth

**`get_n_leaves()`**
- **Purpose**: Get the number of leaf nodes
- **Input**: None
- **Output**: int - Number of leaves

### Algorithm Details

1. **Splitting Criterion**: Uses Gini impurity to find the best split
   - Gini impurity = 1 - Σ(p_i²) where p_i is the proportion of class i
   - Lower Gini = purer node

2. **Split Selection**: For each node:
   - Try all features and all unique threshold values
   - Calculate weighted Gini impurity for each split
   - Select split with lowest weighted Gini

3. **Stopping Criteria**:
   - Maximum depth reached
   - Node contains only one class (pure)
   - Number of samples below min_samples_split
   - Split would violate min_samples_leaf

4. **Prediction**: Traverse tree from root to leaf based on feature values

### Tree Structure

The tree is stored as a nested dictionary:
```python
{
    'leaf': False,           # True if leaf node
    'feature': 2,            # Feature index to split on
    'threshold': 5.3,        # Threshold value
    'left': {...},           # Left subtree (feature <= threshold)
    'right': {...}           # Right subtree (feature > threshold)
}

# Leaf node:
{
    'leaf': True,
    'value': 1               # Predicted class
}
```

### When to Use

- **Pros**:
  - Easy to interpret and visualize
  - Handles both numerical and categorical features
  - No need for feature scaling
  - Can capture non-linear relationships
  - Fast prediction (O(log n) for balanced tree)

- **Cons**:
  - Prone to overfitting (especially with deep trees)
  - Can be unstable (small data changes → large tree changes)
  - Biased toward features with more levels
  - Cannot extrapolate

---

## Random Forest

### Overview
An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees. Reduces overfitting and improves generalization.

### Class: `RandomForest`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees in the forest |
| `max_depth` | int or None | None | Maximum depth of each tree |
| `min_samples_split` | int | 2 | Minimum samples required to split a node |
| `min_samples_leaf` | int | 1 | Minimum samples required at a leaf node |
| `max_features` | int, float, str or None | 'sqrt' | Number of features to consider for best split:<br>- int: use that many features<br>- float: use that fraction of features<br>- 'sqrt': use sqrt(n_features)<br>- 'log2': use log2(n_features)<br>- None: use all features |
| `bootstrap` | bool | True | Whether to use bootstrap samples when building trees |
| `random_state` | int or None | None | Random seed for reproducibility |

#### Methods

**`fit(X, y)`**
- **Purpose**: Build a forest of decision trees
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Training feature matrix
  - `y`: numpy.ndarray of shape `(n_samples,)` - Training target labels (0 or 1)
- **Output**: Returns `self`

**`predict(X)`**
- **Purpose**: Predict class labels using majority voting
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
- **Output**: numpy.ndarray of shape `(n_samples,)` - Predicted class labels (0 or 1)

**`predict_proba(X)`**
- **Purpose**: Predict class probabilities
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
- **Output**: numpy.ndarray of shape `(n_samples, 2)` - Probabilities `[P(class=0), P(class=1)]`

**`score(X, y)`**
- **Purpose**: Calculate accuracy score
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Test samples
  - `y`: numpy.ndarray of shape `(n_samples,)` - True labels
- **Output**: float - Accuracy (fraction of correct predictions)

**`feature_importance(X, y)`**
- **Purpose**: Calculate feature importance (simplified version)
- **Input**:
  - `X`: numpy.ndarray of shape `(n_samples, n_features)` - Training samples
  - `y`: numpy.ndarray of shape `(n_samples,)` - True labels
- **Output**: numpy.ndarray of shape `(n_features,)` - Feature importances (sum to 1)

### Algorithm Details

1. **Bootstrap Aggregating (Bagging)**:
   - For each tree, create a bootstrap sample (random sample with replacement)
   - Each bootstrap sample has same size as original dataset
   - On average, each sample contains ~63.2% unique instances

2. **Random Feature Selection**:
   - At each split, randomly select a subset of features
   - This reduces correlation between trees
   - Default: sqrt(n_features) for classification

3. **Tree Building**:
   - Build each decision tree on its bootstrap sample
   - Use only selected features at each split
   - Trees are grown to full depth (or max_depth if specified)

4. **Prediction**:
   - Get prediction from each tree
   - For classification: use majority voting
   - For probabilities: average the votes across all trees

### When to Use

- **Pros**:
  - Very robust and accurate
  - Reduces overfitting compared to single tree
  - Provides feature importance
  - Handles high-dimensional data well
  - Works with missing values (with modifications)
  - Parallelizable (trees built independently)

- **Cons**:
  - Less interpretable than single tree
  - Slower to train and predict than single tree
  - Larger memory footprint
  - Cannot extrapolate
  - May overfit on noisy datasets

---

## Usage Examples

### Example 1: KNN Classification

```python
import numpy as np
from models import KNN

# Load your fire prediction data
X_train = np.array([...])  # shape: (n_samples, n_features)
y_train = np.array([...])  # shape: (n_samples,) - binary (0 or 1)
X_test = np.array([...])
y_test = np.array([...])

# Create and train KNN model
knn = KNN(k=5, distance_metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Get probabilities
probabilities = knn.predict_proba(X_test)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Decision Tree Classification

```python
from models import DecisionTree

# Create and train decision tree
tree = DecisionTree(max_depth=10, min_samples_split=5, min_samples_leaf=2)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Evaluate
accuracy = tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get tree information
print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
```

### Example 3: Random Forest Classification

```python
from models import RandomForest

# Create and train random forest
rf = RandomForest(
    n_estimators=100,
    max_depth=15,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Get probabilities
probabilities = rf.predict_proba(X_test)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get feature importance
importances = rf.feature_importance(X_train, y_train)
print("Feature importances:", importances)
```

### Example 4: Comparing All Models

```python
from models import KNN, DecisionTree, RandomForest

models = {
    'KNN': KNN(k=5),
    'Decision Tree': DecisionTree(max_depth=10),
    'Random Forest': RandomForest(n_estimators=100, random_state=42)
}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print()
```

---

## Input Data Requirements

### General Requirements

All models expect:

1. **Feature Matrix (X)**:
   - Type: numpy.ndarray
   - Shape: `(n_samples, n_features)`
   - Values: Numerical (int or float)
   - Normalized/scaled: Recommended for KNN, optional for tree-based models

2. **Target Labels (y)**:
   - Type: numpy.ndarray
   - Shape: `(n_samples,)`
   - Values: Binary classification (0 or 1)
   - Should be integers or easily convertible to integers

### Fire Prediction Context

For the fire prediction problem, your input should be:

```python
# Features from different datasets
features = [
    'elevation',      # Elevation features
    'slope',
    'aspect',
    'roughness',
    'tmax',          # Climate features
    'tmin',
    'prec',
    'soil_type',     # Soil features
    'land_cover'     # Land cover features
]

# Target
target = 'class'  # 0 = no fire, 1 = fire
```

### Preprocessing Recommendations

1. **Feature Scaling** (for KNN):
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Handling Missing Values**:
   ```python
   # Remove rows with missing values
   mask = ~np.isnan(X).any(axis=1)
   X = X[mask]
   y = y[mask]
   ```

3. **Train-Test Split**:
   ```python
   # Simple split without sklearn
   n_samples = len(X)
   n_train = int(0.8 * n_samples)
   indices = np.random.permutation(n_samples)
   
   train_idx = indices[:n_train]
   test_idx = indices[n_train:]
   
   X_train, X_test = X[train_idx], X[test_idx]
   y_train, y_test = y[train_idx], y[test_idx]
   ```

---

## Performance Considerations

### Memory Usage

- **KNN**: O(n × m) - stores all training data
- **Decision Tree**: O(nodes × m) - stores tree structure
- **Random Forest**: O(n_estimators × nodes × m) - stores multiple trees

Where:
- n = number of training samples
- m = number of features
- nodes = number of nodes in tree

### Time Complexity

| Model | Training | Prediction (single sample) |
|-------|----------|---------------------------|
| KNN | O(1) | O(n × m) |
| Decision Tree | O(n × m × log n) | O(log nodes) |
| Random Forest | O(n_estimators × n × m × log n) | O(n_estimators × log nodes) |

### Recommendations

1. **Small Dataset (< 1000 samples)**:
   - All models will work well
   - Decision Tree or KNN for interpretability

2. **Medium Dataset (1000-10000 samples)**:
   - Random Forest for best accuracy
   - Decision Tree for speed and interpretability

3. **Large Dataset (> 10000 samples)**:
   - Random Forest (with fewer estimators)
   - Avoid KNN (too slow)

4. **High-Dimensional Data (many features)**:
   - Random Forest with feature selection
   - Avoid KNN (curse of dimensionality)

---

## Model Selection Guidelines

| Criterion | KNN | Decision Tree | Random Forest |
|-----------|-----|---------------|---------------|
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed (Training) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Speed (Prediction) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Interpretability | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Handles Overfitting | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Feature Scaling Required | Yes | No | No |
| Works with Missing Data | No | Yes* | Yes* |
| Memory Usage | High | Low | High |

*With modifications (not implemented in basic version)

---

## Troubleshooting

### Common Issues

1. **"Arrays have different shapes"**
   - Ensure X and y have compatible shapes
   - Check that n_samples is consistent

2. **Poor KNN performance**
   - Features may not be scaled
   - k value may be too high or too low
   - Try different distance metrics

3. **Decision Tree overfitting**
   - Increase min_samples_split
   - Decrease max_depth
   - Increase min_samples_leaf

4. **Random Forest too slow**
   - Reduce n_estimators
   - Reduce max_depth
   - Use max_features='sqrt'

5. **Out of memory errors**
   - Reduce n_estimators for Random Forest
   - Subsample your data
   - Use max_depth to limit tree size

---

## References

- **KNN**: Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification
- **Decision Trees**: Breiman, L., et al. (1984). Classification and Regression Trees (CART)
- **Random Forest**: Breiman, L. (2001). Random Forests, Machine Learning

---

## License

These implementations are for educational purposes. Use at your own risk in production environments.
