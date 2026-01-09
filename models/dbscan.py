import numpy as np

class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implemented from scratch.
    
    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered as a core point
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan')
    precompute : bool, default=True
        If True, pre-calculate distance matrix for faster clustering (O(n²) space, 10-50x speedup)
        If False, calculate distances on-the-fly (O(1) space, slower)
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', precompute=True):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.precompute = precompute
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        self.distance_matrix_ = None
        
    def _compute_distance(self, x1, x2):
        """Compute distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_distance_matrix(self, X):
        """
        Pre-calculate all pairwise distances.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix
        """
        n_samples = X.shape[0]
        
        if self.metric == 'euclidean':
            # Vectorized euclidean distance using broadcasting
            # Distance = sqrt(sum((xi - xj)^2)) = sqrt(|xi|^2 + |xj|^2 - 2*xi·xj)
            sq_sum = np.sum(X ** 2, axis=1, keepdims=True)  # shape: (n_samples, 1)
            distances_sq = sq_sum + sq_sum.T - 2 * np.dot(X, X.T)
            # Clip negative values (due to numerical errors) to 0
            distances_sq = np.maximum(distances_sq, 0)
            self.distance_matrix_ = np.sqrt(distances_sq)
            
        elif self.metric == 'manhattan':
            # Manhattan distance
            self.distance_matrix_ = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dist = np.sum(np.abs(X[i] - X[j]))
                    self.distance_matrix_[i, j] = dist
                    self.distance_matrix_[j, i] = dist
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_neighbors(self, X, point_idx):
        """
        Find all neighbors within eps distance of a point.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix (used only if distance matrix not precomputed)
        point_idx : int
            Index of the point to find neighbors for
            
        Returns
        -------
        neighbors : list
            Indices of neighboring points
        """
        if self.distance_matrix_ is not None:
            # Fast: O(n) lookup from pre-calculated matrix
            neighbors = np.where(self.distance_matrix_[point_idx] <= self.eps)[0].tolist()
        else:
            # Slow: O(n*d) on-the-fly calculation
            neighbors = []
            point = X[point_idx]
            
            for idx, other_point in enumerate(X):
                if self._compute_distance(point, other_point) <= self.eps:
                    neighbors.append(idx)
        
        return neighbors
    
    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expand a cluster from a core point.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
        labels : numpy.ndarray
            Current cluster labels
        point_idx : int
            Index of the core point
        neighbors : list
            Initial neighbors of the core point
        cluster_id : int
            ID of the cluster being expanded
        """
        # Add seed point to cluster
        labels[point_idx] = cluster_id
        
        # Use a queue to process neighbors
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor is noise, add to cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited
            elif labels[neighbor_idx] == -2:  # -2 means unvisited
                labels[neighbor_idx] = cluster_id
                
                # Find neighbors of this neighbor
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # If this neighbor is also a core point, add its neighbors to the queue
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend([n for n in neighbor_neighbors if n not in neighbors])
            
            i += 1
    
    def fit(self, X):
        """
        Fit DBSCAN clustering.
        
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
        n_samples = X.shape[0]
        
        # Pre-calculate distance matrix if requested
        if self.precompute:
            print(f"[DBSCAN] Pre-computing distance matrix ({n_samples} samples)...")
            self._compute_distance_matrix(X)
            matrix_memory_mb = (n_samples ** 2 * 8) / (1024 ** 2)
            print(f"[DBSCAN] Distance matrix computed: shape {self.distance_matrix_.shape}, memory: {matrix_memory_mb:.1f} MB")
        
        # Initialize all points as unvisited (-2)
        labels = np.full(n_samples, -2, dtype=int)
        
        # Track core samples
        core_samples = []
        
        cluster_id = 0
        
        # Process each point
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != -2:
                continue
            
            # Find neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                # Mark as noise (will be updated if later added to cluster)
                labels[point_idx] = -1
            else:
                # Core point - expand cluster
                core_samples.append(point_idx)
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
                cluster_id += 1
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = len(np.unique(labels[labels >= 0]))
        
        return self
    
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
            Cluster labels (-1 for noise points)
        """
        self.fit(X)
        return self.labels_
    
    def get_core_samples(self):
        """
        Get indices of core samples.
        
        Returns
        -------
        core_samples : numpy.ndarray
            Indices of core samples
        """
        if self.core_sample_indices_ is None:
            raise ValueError("Model must be fitted first")
        return self.core_sample_indices_
    
    def get_noise_samples(self):
        """
        Get indices of noise samples.
        
        Returns
        -------
        noise_samples : numpy.ndarray
            Indices of noise samples (outliers)
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
        return np.where(self.labels_ == -1)[0]
