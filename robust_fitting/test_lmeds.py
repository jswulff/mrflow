#! /usr/bin/env python2

import numpy as np
from matplotlib import pyplot as plt

import lmeds

a1 = -5
a2 = 0.2

x_ = np.random.randn(100,1)*10
y_ = a1 * x_ + a2

x = x_ + np.random.rand(100,1) * 5 - 1
y = y_ + np.random.rand(100,1) * 5 - 1

A = np.c_[x, np.ones_like(x)]
b = y.copy()
b[-48:] *= -1

model_lstsq = np.linalg.lstsq(A,b)[0]
model_lmeds = lmeds.solve(A,b)[0]
model_lmeds_norecomp = lmeds.solve(A,b,recompute_model=False)[0]


xin = np.linspace(-10.0,10.0,1000)
l1 = xin * model_lstsq[0] + model_lstsq[1]
l2 = xin * model_lmeds[0] + model_lmeds[1]
l3 = xin * model_lmeds_norecomp[0] + model_lmeds_norecomp[1]
l4 = xin * a1 + a2

print(xin.shape)
print(l1.shape)
print(l2.shape)
print(l3.shape)
print(l4.shape)

plt.figure()
plt.scatter(x,b)
plt.plot(xin,l1,label='LS')
plt.plot(xin,l2,label='LM, recomputed')
plt.plot(xin,l3,label='LM, no recomputed')
plt.plot(xin, l4, label='True line')
plt.legend()
plt.show()

