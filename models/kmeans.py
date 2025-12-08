import numpy as np

class KMeans:
    """
    K-Means clustering algorithm implemented from scratch.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form
    max_iters : int, default=100
        Maximum number of iterations for the algorithm
    tol : float, default=1e-4
        Tolerance for convergence (change in centroids)
    random_state : int or None, default=None
        Random seed for reproducibility
    init : str, default='k-means++'
        Initialization method ('random' or 'k-means++')
    """
    
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """Initialize centroids using random or k-means++ method."""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        if self.init == 'random':
            # Randomly select k samples as initial centroids
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[random_indices].copy()
        
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = []
            
            # Choose first centroid randomly
            first_idx = np.random.randint(n_samples)
            centroids.append(X[first_idx])
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Compute distances from each point to nearest existing centroid
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids])
                    for x in X
                ])
                
                # Choose next centroid with probability proportional to distance squared
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                
                for idx, cum_prob in enumerate(cumulative_probs):
                    if r < cum_prob:
                        centroids.append(X[idx])
                        break
            
            return np.array(centroids)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X):
        """Assign each sample to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as the mean of assigned samples."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute inertia (within-cluster sum of squares)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means clustering.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        X = np.array(X)
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterative optimization
        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        # Final assignment
        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for samples.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Cluster labels for each sample
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Fit and predict clusters in one step.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Cluster labels for each sample
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X):
        """
        Transform X to cluster-distance space.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Samples to transform
            
        Returns
        -------
        distances : numpy.ndarray of shape (n_samples, n_clusters)
            Distance to each cluster centroid
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before transformation")
        
        X = np.array(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return distances
