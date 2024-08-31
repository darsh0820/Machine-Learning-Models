import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters

    def fit(self, x, y):
        self.theta = np.zeros(x.shape[1])

        for i in range(self.num_iters):
            prediction = np.dot(x, self.theta)
            
            gradient = np.dot(x.T, (prediction - y)) / y.size
            
            self.theta -= self.lr * gradient

            if i % 100 == 0:
                cost = np.mean((prediction - y) ** 2) / 2
                print(f'Iteration {i}: Cost {cost}')

    def predict(self, x):
        return np.dot(x, self.theta)
