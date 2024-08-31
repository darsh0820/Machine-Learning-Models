import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.01,num_iters=1000):
        self.lr=lr
        self.num_iters=num_iters

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,x,y):
        self.theta = np.zeros(x.shape[1])

        for i in range (self.num_iters):
            z = np.dot(x,self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(x.T,(h-y))/y.size
            self.theta -= self.lr * gradient

            if i % 100 == 0:
                cost = -y.dot(np.log(h)) - (1-y).dot(np.log(1-h))
                print(f'Iteration {i}: Cost {cost}')

    def predict_prob(self,x):
        return self.sigmoid(np.dot(x,self.theta))
    
    def predict(self, x, threshold = 0.5):
        return self.predict_prob(x) >= threshold