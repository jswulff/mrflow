#! /usr/bin/env python2

import numpy as np

import ransac

from matplotlib import pyplot as plt

a = -3.2
b = 7.5

# Generate random data
X = np.random.rand(1000) * 10 - 5.0
Y = a * X + b
Y += (np.random.rand(1000) * 2.0 - 1.0)
Y[-200:] = (np.random.rand(200) * 20.0 - 10.0)

def estimate_model(X):
    x = X[:,0]
    y = X[:,1]

    A = np.c_[x,np.ones_like(x)]
    b = y
    model = np.linalg.lstsq(A,b)[0]
    return model

def estimate_inliers(X, model):
    y_hat = model[0] * X[:,0] + model[1]
    inliers = np.abs(X[:,1]-y_hat) < 1.0
    return inliers

best_model, best_inliers = ransac.estimate(np.c_[X,Y], 2, estimate_model, estimate_inliers,p_outlier=0.8)

print('Best model: {}'.format(best_model))

plt.figure()
plt.scatter(X[best_inliers],Y[best_inliers],color='blue')
plt.scatter(X[best_inliers==0],Y[best_inliers==0],color='red')

x_ = np.linspace(X.min(),X.max(),100)
y_ = x_ * a + b
y_hat = x_ * best_model[0] + best_model[1]

plt.plot(x_,y_,label='True fit')
plt.plot(x_,y_hat,label='Best fit')

plt.legend()
plt.show()


