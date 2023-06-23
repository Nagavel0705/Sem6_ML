import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=3, max_iters=100, plot_steps = True):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # Initialising 'K' number of empty clusters
        self.clusters = [[] for i in range(self.K)] 

        # Initialising the centroids list
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Assign the centroids randomly
        random_indxs = np.random.choice(self.n_samples, size=self.K, replace=False)
        self.centroids = [self.X[indx] for indx in random_indxs]

        # Optimize clusters (Improving accuracy)
        for i in range(self.max_iters):
            
            # Create clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot(self.n_features)

            # Calculate new centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence
            if self._is_converged(old_centroids, self.centroids):
                break

            if self.plot_steps:
                self.plot(self.n_features)

        return self._get_cluster_labels(self.clusters), self.centroids
    
    def _get_cluster_labels(self, clusters):
        
        # Assigning cluster numbers to samples
        labels = np.zeros(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            # print("cluster index = ", cluster_idx, "cluster = ", cluster)
            for sample_index in cluster:
                # print("sample index = ", sample_index)
                labels[sample_index] = cluster_idx
                # print("labels = ", labels)

        return labels
    
    def _create_clusters(self, centroids):
        clusters = [[] for i in range(self.K)]

        for idx, sample in enumerate(self.X):
            centroid_indx = self._closest_centroid(sample, centroids)
            clusters[centroid_indx].append(idx)
        
        return clusters 
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)

        return closest_centroid

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        
        # print("In computation centroids = ", centroids)
        return centroids
    
    def _is_converged(self, old_centroids, new_centroids):
        return np.array_equal(old_centroids, new_centroids)

    def plot(self, no_of_features):

        # As no. of features determine the number of dimensions 

        # Plotting for 2D graphs 
        if no_of_features == 2:
            fig, ax = plt.subplots(figsize=(12,8))
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        # Plotting for 3D graphs
        elif no_of_features == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()

if __name__ == "__main__":

    X, y, centers = make_blobs(
        centers=3, n_samples=1000, n_features=2, shuffle=True, random_state=70, return_centers=True
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    y_pred, centroids = k.predict(X)
    print("y_pred = ", y_pred)
    print("y = ", y)
    print("\nKMeans centroids = ", centroids)
    print("\nActual centroids = ", centers)
    k.plot(X.shape[1])