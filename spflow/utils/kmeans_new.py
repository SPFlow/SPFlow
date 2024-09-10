import torch
import torch.nn as nn

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32)

        if self.random_state:
            torch.manual_seed(self.random_state)

        # Randomly initialize the centroids
        random_indices = torch.randperm(X.size(0))[:self.n_clusters]
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Assign clusters based on closest centroids
            distances = torch.cdist(X, self.centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            # Calculate new centroids
            new_centroids = torch.stack([X[labels == j].mean(dim=0) for j in range(self.n_clusters)])

            # Check for convergence
            if torch.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        distances = torch.cdist(X, self.centroids, p=2)
        return torch.argmin(distances, dim=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def inertia(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        labels = self.predict(X)
        distances = torch.cdist(X, self.centroids, p=2)
        return torch.sum(distances[torch.arange(X.size(0)), labels])

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import numpy as np

    # Generate a dataset
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Fit and predict using the custom KMeans
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    print("Cluster centroids:")
    print(kmeans.centroids)
    print("Predicted labels:")
    print(labels)
