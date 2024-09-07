import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.c_[np.ones(X.shape[0]), X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(lr=0.01, num_iters=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')