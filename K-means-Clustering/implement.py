from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from KMeansClustering import KMeans
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(k=3, num_iters=200)
kmeans.fit(X)

predictions = kmeans.predict(X)

accuracy = accuracy_score(y,predictions)
print(f'Accuracy: {accuracy:.2f}')