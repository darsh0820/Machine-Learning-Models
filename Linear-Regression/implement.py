import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression 

data = fetch_california_housing()  
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.c_[np.ones(X.shape[0]), X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(lr=0.01, num_iters=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = np.mean((predictions - y_test) ** 2)
print(f'Test Mean Squared Error: {mse:.2f}')
