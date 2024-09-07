import numpy as np

class KMeans:
    def __init__(self, k=3, num_iters=100):
        self.k = k
        self.num_iters = num_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for i in range(self.num_iters):
            self.labels = self._assign_clusters(X)
            
            old_centroids = self.centroids
            self.centroids = self._update_centroids(X)

            cost = self._calculate_cost(X)
            print(f'Iteration {i+1}: Cost {cost}')

            if np.all(old_centroids == self.centroids):
                break

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

    def _calculate_cost(self, X):
        cost = 0
        for i in range(self.k):
            cost += np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
        return cost

    def predict(self, X):
        return self._assign_clusters(X)
