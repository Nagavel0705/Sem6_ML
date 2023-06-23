import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib.colors import ListedColormap

def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict(x_test) for x_test in X_test]
        return predictions

    def _predict(self, x_test):
        # Calculate distances of the testing point from the training data points
        distances = [euclidean_distance(x_test,x_train) for x_train in self.X_train]

        # Sort distances in ascending order
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        print("K nearest indices and labels = ", k_nearest_indices, ": ", k_nearest_labels)

        most_common = Counter(k_nearest_labels).most_common()
        print("Most common = ", most_common)
        return most_common[0][0]
    

if __name__ == "__main__":

    cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # print(X[:,0])
    # print(X[:,3])
    plt.figure()
    plt.scatter(X[:,1], X[:,2], c=y, cmap=cmap, s=20)
    plt.show()

    classify = KNN(k=5)
    classify.fit(X_train,y_train)
    predictions = classify.predict(X_test)

    print(predictions)

    accuracy = np.sum(predictions == y_test) / len(y_test)

    print(accuracy)
