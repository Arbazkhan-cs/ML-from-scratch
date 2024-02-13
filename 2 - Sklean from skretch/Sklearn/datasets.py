#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def generate_linear_dataset(num_samples=100, num_features=1, noise=0.1):
        # Generate random features
        X = np.random.rand(num_samples, num_features)

        # Generate true coefficients for the linear model
        true_coefficients = np.random.rand(num_features)

        # Generate target variable with noise
        y = np.dot(X, true_coefficients) + np.random.normal(0, noise, num_samples)

        return X, y
    
def generate_sigmoid_dataset(num_samples=100, num_features=1, noise=0.1):
        # Generate random features
        X = np.random.randn(num_samples, num_features)

        # Generate true coefficients for the logistic model
        true_coefficients = np.random.randn(num_features)

        # Generate logits (z) without noise
        logits = np.dot(X, true_coefficients)

        # Apply logistic function to convert logits to probabilities
        probabilities = 1 / (1 + np.exp(-logits))

        # Generate binary labels based on probabilities with added noise
        y = np.random.binomial(n=1, p=probabilities)  # Add noise to convert probabilities to binary labels

        return X, y