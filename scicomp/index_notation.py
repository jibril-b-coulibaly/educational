# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:16:47 2020

@author: jibri
"""

import numpy as np

# Computes the Jacobian in row-order expansion of dE/dF, E=1/2(F^t * F - I)
# \frac{\partial E_{ij}}{\partial F_{kl}} = \frac{1}{2}\left(F_{kj} \delta_{li} + F_{ki} \delta_{lj}\right)
results = np.zeros(81,dtype=object)
n = 0
for i in range(1,4):
    for j in range(1,4):
        for k in range(1,4):
            for l in range(1,4):
                string = ''
                if (l==i):
                    string = string+'F'+str(k)+str(j)
                if (l==j):
                    if(l==i):
                        string = string+' + F'+str(k)+str(i)
                    else:
                        string = string+'F'+str(k)+str(i)
                if (l!= i and l!=j):
                    string='0'
                results[n] = string
                n = n+1
                