import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression 


# Load the Boston Housing dataset (deprecated in sklearn, replace with similar dataset if needed)
data = fetch_california_housing()  
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add intercept term (bias)
X = np.c_[np.ones(X.shape[0]), X]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression(lr=0.01, num_iters=1000)
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate Mean Squared Error as a metric for evaluation
mse = np.mean((predictions - y_test) ** 2)
print(f'Test Mean Squared Error: {mse:.2f}')
