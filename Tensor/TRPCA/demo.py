# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:16:54 2019

@author: M.Sc. Fernando Hermosillo
"""

import numpy as np
import trpca

# Create a sintetic 3rd-order tensor data
L=np.ones((4,5,3))*5
S=np.random.rand(4,5,3)
X = L + S

# Decompose the tensor into a low-rank and sparse tensor
Lhat,Shat,obj,err,itercount=trpca.trpca_tnn(X)

# Print out results
print('Low-rank tensor was recovered in', itercount, 'iterations, with an error of', err)
print('hat(L):')
trpca.tprint(Lhat)
print('hat(S):')
trpca.tprint(Shat)
