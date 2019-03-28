# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:16:54 2019

@author: ferna
"""

import numpy as np
import trpca

A=np.random.rand(3,4,3)
A[:,:,0]=[[1,2,3,4], [5,6,7,8], [9,10,11,12]]
A[:,:,1]=[[13,14,15,16],[17,18,19,20],[21,22,23,24]]
A[:,:,2]=[[25,26,27,28],[29,30,31,32],[33,34,35,36]]
L,S,obj,err,itercount=trpca.trpca_tnn(A)


trpca.tprint(L)
trpca.tprint(S)
print('Error:', err)
print('Iterations:', itercount)

#L,S,obj,err,iterations=trpca_tnn(A,lambda_)

#print('Error:', err, 'with', iterations, 'iterations')
#tprint(L)
#tprint(S)