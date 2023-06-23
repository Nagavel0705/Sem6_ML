import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def plot(clusters, centroids, no_of_features):

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

        for i, index in enumerate(clusters):
            point = X[index].T
            ax.scatter(*point)

        for point in centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()

if __name__ == "__main__":

    X, y, centers = make_blobs(
        centers=3, n_samples=1000, n_features=2, shuffle=True, random_state=70, return_centers=True
    )
    print(X.shape)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

    #predict the labels of clusters.
    label = kmeans.fit_predict(X)

    print(kmeans.cluster_centers_)
    print("\nActual centroids = ", centers)

    test_point = [8,8]
    min_dist = sys.maxsize
    for i in range(X.shape[1]):
        if euclidean_distance(test_point, kmeans.cluster_centers_[i]) <= min_dist:
            min_dist = euclidean_distance(test_point, kmeans.cluster_centers_[i])
            classified_into = i
    print("Classified into:", classified_into)

    #Getting unique labels
 
    u_labels = np.unique(label)
    
    #plotting the results:
    
    for i in u_labels:
        plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
    plt.scatter(test_point[0], test_point[1], color='black', label='New Point')
    plt.legend()
    plt.show()