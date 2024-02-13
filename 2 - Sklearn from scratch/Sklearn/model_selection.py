#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y, alpha=0.01, iteration=10000):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iteration = iteration
        self.coef__ = np.zeros(X.shape[1])
        self.intercept__ = 0


    def function(self):
        return np.dot(self.X, self.coef__) + self.intercept__

    def compute_cost(self):
        f_wb = self.function()
        m = len(self.X)
        cost = np.sum((f_wb - self.y)**2)
        return (1/(2*m))*cost

    def fit(self):
        X_scale = (self.X-np.mean(self.X))/np.std(self.X)
        m = len(self.X)
        prevCost = 0

        for i in range(self.iteration):
            y_hat = np.dot(X_scale, self.coef__) + self.intercept__
            error = y_hat - self.y

            d_dw = (1/m)*np.dot(X_scale.T, error)
            d_db = (1/m)*np.sum(error)

            self.coef__ -= self.alpha * d_dw
            self.intercept__ -= self.alpha * d_db

            cost = self.compute_cost()              

            if (prevCost == cost) and i>0:
                break
            prevCost = cost

        return self
    

    def predict(self, X_test):
        X_test = (X_test-np.mean(X_test))/np.std(X_test)
        return np.dot(X_test, self.coef__) + self.intercept__

    def bestFitLine(self):
        y_hat = self.predict(self.X)
        plt.scatter(self.X[:,0], self.y, label="Data Points")
        plt.plot(self.X[:, 0], y_hat, color="orange", label="Linear Line")
        plt.legend()
        plt.grid(True)
        plt.show()



class LogisticRegression:
    def __init__(self, X, Y, alpha=0.01, iteration=10000):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.iteration = iteration
        self.coef__ = np.zeros(X.shape[1])
        self.intercept__ = 0
    
    def function(self, X):
        z = np.dot(X, self.coef__) + self.intercept__
        gz = 1/(1+(np.exp(-z)))
        return gz
    
    def compute_cost(self, X):
        m = len(X)
        gz = self.function(X)
        loss = (-self.Y*(np.log(gz))) - ((1 - self.Y)*np.log(1 - gz))
        return (1/m)*np.sum(loss)
    
    def fit(self):
        m = len(self.X)
        X_scale = (self.X-np.mean(self.X))/np.std(self.X)
        prev_cost = 0
        
        for i in range(self.iteration):
            gz = self.function(X_scale)
            error = gz - self.Y
            
            d_dw = (1/m)*np.dot(X_scale.T, error)
            d_db = (1/m)*np.sum(error)
            
            self.coef__ -= self.alpha * d_dw
            self.intercept__ -= self.alpha * d_db
            
            cost = self.compute_cost(X_scale)
            
            if np.abs(prev_cost - cost) < 1e-6:
                break
            
            prev_cost = cost
        
        return self
        
    def predict(self, X_test):
        X_test = (X_test-np.mean(X_test))/np.std(X_test)
        y_hat = self.function(X_test)
        return y_hat

    def bestFitLine(self):
        y_hat = self.predict(self.X)
        plt.scatter(self.X[:,0], self.Y, label="Data Points")
        plt.plot(self.self.X[:, 0], y_hat, color="orange", label="Decision Boundary")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_decision_boundary(self):
        # Plot data points
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        # Plot decision boundary
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                               np.arange(x2_min, x2_max, 0.1))

        Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        plt.contour(xx1, xx2, Z, levels=[0.5], colors='orange', linestyles='--')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

