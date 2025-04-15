# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:50:44 2025

@author: JibrilCoulibaly
"""

import numpy as np
from matplotlib import pyplot as plt



def cov_mat(x, y, func):
    mat = np.zeros((len(x),len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            mat[i,j] = func(xi, yj)
    return mat


# Sample path for zero mean GP

# Select times
ndata = 200
t = np.linspace(0, 10, ndata)

# Mean
mu = np.zeros(ndata)



# covariance matrix
scale = 1
cov_func = lambda xi,xj: np.exp(-0.5*(xi-xj)**2/(scale*scale))
cov = cov_mat(t, t, cov_func)

# Normal Random Vector Generation
rng = np.random.default_rng()
X = rng.multivariate_normal(mu, cov)

plt.plot(t, X, 'b', label='realization')








# Kriging

# Random realization from before
# Select points
idx = [10, 20, 70, 100, 160]
t_observed = t[idx]
x_observed = X[idx]
plt.plot(t_observed, x_observed, marker='o', linestyle='none', color='r', label='observations')



cov_observed = cov_mat(t_observed, t_observed, cov_func)
cov_vector = cov_mat(t_observed, t, cov_func)
weights = np.linalg.solve(cov_observed, cov_vector)
X_estimate = np.dot(x_observed, weights)
var_diff = cov_func(t, t) - np.sum(cov_vector * weights, axis=0)

plt.plot(t, X_estimate, color='g', label='Kriging')
plt.fill_between(t, X_estimate-var_diff, X_estimate+var_diff, color='g', alpha=0.2)

plt.legend()
plt.show()


