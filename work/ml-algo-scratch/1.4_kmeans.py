import numpy as np
import matplotlib.pyplot as plt

import numpy as np
class Kmean:
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
        self.centroid = []
        
    def fit(self, X, y):

        centroid = X[np.random.choice(range(X.shape[0]), size=self.k, replace = True).astype(int)]

        for i in range(self.iterations):
            distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis = -1)
            
            #np.argmin, the parameter axis=-1 indicates that the minimum values 
            #should be computed along the last axis of the input array
            #[[8.60232527 1.        ]
            # [7.81024968 3.16227766]
            # [9.43398113 0.        ]
            # [2.         7.81024968]
            # [1.41421356 8.06225775]
            # [0.         9.43398113]]
            # pick minimum values from last axis and give 1 if minimum otherwise 0
            # [1, 1, 1, 0, 0, 0]
            
            labels = np.argmin(distances, axis=-1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.k)])
        
        self.centroid = new_centroids
        return new_centroids, labels
    
    def predict(self, X):
        X = np.asarray(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroid, axis=-1)
        return np.argmin(distances, axis=-1)

np.random.seed(10)            
kmean = Kmean(3, 1000)

X = np.array([[1,10], [3,8], [4,2], [50,51], [52,54], [55,51], [200,209]])
y = [0,0,0,1,1,1,2]

centroids, labels = kmean.fit(X, y)


print(kmean.predict([[51,56]]))


# Plotting the data points and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
