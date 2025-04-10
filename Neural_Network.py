import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layers, alpha = 0.1):
        self.layers = layers
        self.alpha = alpha
        self.bias = []
        self.w = []
        
        for i in range(0, len(layers) - 1):
            w_ = np.random.rand(layers[i - 1], layers[i]) * np.sqrt(2.0 / layers[i])
            bias_ = np.zeros((layers[i + 1], 1))
            self.w.append(w_)
            self.bias.append(bias_)
    
    def sigmoid(self, X):
        return 1.0/(1 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        return X/(1 + X)
    
    def forward(self, X):
        self.A = [X]
        out = X 
        
        for i in range(len(self.layers) - 1):
            out = self.sigmoid(np.dot(out, self.w[i]) + self.bias[i])
            self.A.append(out)
        return out
    
    def backward(self, X, y):
        self.A = [X]
        y = y.reshape(-1, 1)
        dA = [-(y/self.A[-1] - (1 - y)/(1 - self.A[-1]))]
        dw = []
        db = []
        for i in reversed(range(0, len(self.layers) - 1)):
            dw_ = np.dot(self.A[i].T, (dA[-1] * self.sigmoid_derivative(self.A[i + 1])))
            db_ = (np.sum(dA[-1] * self.sigmoid_derivative(self.A[i + 1]), 0)).reshape(-1, 1)
            dA_ = (dA[-1] * self.sigmoid_derivative(np.dot(self.A[i + 1]), self.w[i].T))
            dw.append(dw_)
            db.append(db_)
            dA.append(dA_)
        dw = dw[::-1]
        db = db[::-1]
        for i in range(len(self.layers) - 1):
            self.w[i] = self.w[i] - self.alpha * dw[i]
            self.bias[i] = self.bias[i] - self.alpha * db[i]

    def fit(self, X, y, epochs = 1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = self.calculate_loss(X, y)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)
    
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))