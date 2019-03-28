# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:06:56 2019

@name: TRPCA module
@author: M.Sc. Fernando Hermosillo
"""


# MODULE IMPORTS
import numpy as np



# The tensor tubal rank of a 3 way tensor
#
# X     -    n1*n2*n3 tensor
# trank -    tensor tubal rank of X
#
# version 2.0 - 14/06/2018
#
# Written by Canyi Lu (canyilu@gmail.com)
# Ported by Fernando Hermosillo
#
# References: 
# Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
# June, 2018. https://github.com/canyilu/tproduct.
#
# Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
# Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
# Norm, arXiv preprint arXiv:1804.03728, 2018
#
def tubalrank(X, tol):
	X = np.fft.fft(X,axis=2)
	n1,n2,n3 = X.shape
	s = np.zeros((np.min([n1,n2]),1))

	# i=0
	s = s + np.linalg.svd(X[:,:,0], full_matrices=False)
	
	# i=1,...,halfn3
	halfn3 = np.round(n3/2);
	for i in range(1,halfn3):
		s = s + np.linalg.svd(X[:,:,i], full_matrices=False)*2
	
	# if n3 is even
	if np.mod(n3,2) == 0:
		i = halfn3
		s = s + np.linalg.svd(X[:,:,i], full_matrices=False)
	s = s/n3

	# Check for this line
	#if nargin==1
	#tol = np.max([n1,n2]) * eps(np.max(s));
	trank = np.sum(s[s > tol])

	return trank;

# Tensor-tensor product of two 3 way tensors: C = A*B
# A - n1*n2*n3 tensor
# B - n2*l*n3  tensor
# C - n1*l*n3  tensor
#
# version 2.0 - 09/10/2017
#
# Written by Canyi Lu (canyilu@gmail.com)
# Ported by Fernando Hermosillo
#
# References: 
# Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
# June, 2018. https://github.com/canyilu/tproduct.
#
# Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
# Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
# Norm, arXiv preprint arXiv:1804.03728, 2018
#
def tprod(A,B):	
	n1,n2,n3 = A.shape
	m1,m2,m3 = B.shape
	
	if n2 != m1 or n3 != m3:
		raise ValueError("Inner tensor dimensions must agree.")

	Af = np.fft.fft(A,axis=2);
	Bf = np.fft.fft(B,axis=2);
	Cf = np.zeros((n1,m2,n3),dtype=complex);

	# first frontal slice
	Cf[:,:,0] = Af[:,:,0].dot(Bf[:,:,0])
	
	# i=2,...,halfn3
	halfn3 = int(np.round(n3/2));
	#print("halfn3: ", halfn3)
	for i in range(1,halfn3):
		Cf[:,:,i] = Af[:,:,i].dot(Bf[:,:,i])
		Cf[:,:,n3-i] = np.conj(Cf[:,:,i]); # CHECK INDEXING
		#print("i: ", i, ", n3-i: ", n3-i)

	# if n3 is even
	if np.mod(n3,2) == 0:
		i = halfn3;
		#print("Even: ", i)
		Cf[:,:,i] = Af[:,:,i].dot(Bf[:,:,i])
		
	C = np.fft.ifft(Cf,axis=2);
	
	return C,Af,Bf,Cf


# The proximal operator of the tensor nuclear norm of a 3 way tensor
#
# min_X rho*||X||_* + 0.5*||X-Y||_F^2
#
# Y     -    n1*n2*n3 tensor
#
# X     -    n1*n2*n3 tensor
# tnn   -    tensor nuclear norm of X
# trank -    tensor tubal rank of X
#
# version 2.1 - 14/06/2018
#
# Written by Canyi Lu (canyilu@gmail.com)
# Ported by Fernando Hermosillo
#
# References: 
# Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University. 
# June, 2018. https://github.com/canyilu/tproduct.
#
# Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
# Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
# Norm, arXiv preprint arXiv:1804.03728, 2018
#
def prox_tnn(Y,rho):
    n1,n2,n3 = Y.shape
    X = np.zeros(Y.shape,dtype=complex)
    Y = np.fft.fft(Y,axis=2)
    tnn = 0
    trank = 0
	
    # first frontal slice
    U,S,V = np.linalg.svd(Y[:,:,0], full_matrices=False)
    r = len(S[S>rho])
    if r >= 1:
        S = S[0:r]-rho
        X[:,:,0] = U[:,0:r].dot(np.diag(S)).dot(V.T[:,0:r].T)
        tnn = tnn + np.sum(S)
        trank = np.max([trank,r])
        
    # i=2,...,halfn3
    halfn3 = int(np.round(n3/2));
    for i in range(1,halfn3):
        U,S,V = np.linalg.svd(Y[:,:,i], full_matrices=False)
        r = len(S[S>rho])
        if r >= 1:
            S = S[0:r]-rho;
            X[:,:,i] = U[:,0:r].dot(np.diag(S)).dot(V.T[:,0:r].T)
            tnn = tnn + np.sum(S)*2;
            trank = np.max([trank,r]);
        X[:,:,n3-i] = np.conj(X[:,:,i]);
    
    # if n3 is even
    if np.mod(n3,2) == 0:
        U,S,V = np.linalg.svd(Y[:,:,halfn3], full_matrices=False)
        r = len(S[S>rho])
        if r >= 1:
            S = S[0:r]-rho;
            X[:,:,halfn3] = U[:,0:r].dot(np.diag(S)).dot(V.T[:,0:r].T)
            tnn = tnn + np.sum(S);
            trank = np.max([trank,r]);
    
    # Output results
    tnn = tnn/n3;
    X = np.fft.ifft(X,axis=2);
    
    return X,tnn,trank


    


# The proximal operator of the l1 norm
# 
# min_x lambda*||x||_1 + 0.5*||x-b||_2^2
#
# version 1.0 - 18/06/2016
#
# Written by Canyi Lu (canyilu@gmail.com)
# Ported by Fernando Hermosillo
# 
def prox_l1(b,lambda_):
	import numpy as np
	
	return np.maximum(0,b-lambda_) + np.minimum(0,b+lambda_)







# Solve the Tensor Robust Principal Component Analysis (TRPCA) based on 
# Tensor Nuclear Norm (TNN) problem by ADMM:
#
# min_{L,S} ||L||_*+lambda*||S||_1, s.t. X=L+S
#
# ---------------------------------------------
# Input:
#       X       -    d1*d2*d3 tensor
#       lambda  -    > 0, parameter
#       opts    -    Structure value in Matlab. The fields are
#           opts.tol        -   termination tolerance
#           opts.max_iter   -   maximum number of iterations
#           opts.mu         -   stepsize for dual variable updating in ADMM
#           opts.max_mu     -   maximum stepsize
#           opts.rho        -   rho>=1, ratio used to increase mu
#           opts.DEBUG      -   0 or 1
#
# Output:
#       L       -    d1*d2*d3 tensor
#       S       -    d1*d2*d3 tensor
#       obj     -    objective function value
#       err     -    residual 
#       iter    -    number of iterations
#
# version 1.0 - 19/06/2016
#
# Written by Canyi Lu (canyilu@gmail.com)
# Ported by Fernando Hermosillo
# 
# References: 
# [1] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
#     Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
#     Norm, arXiv preprint arXiv:1804.03728, 2018
# [2] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
#     Yan, Tensor Robust Principal Component Analysis: Exact Recovery of Corrupted 
#     Low-Rank Tensors via Convex Optimization, arXiv preprint arXiv:1804.03728, 2018
#
def trpca_tnn(X,lambda_=np.nan,opts=[]):
    import numpy as np
    # Options structure
    #Options = namedtuple("Options", "tol max_iter rho mu max_mu DEBUG")
    #result = namedtuple('Result',result._fields+('point',))
    
    # Default options
    tol = 1e-8; 
    max_iter = 500; #500
    rho = 1.1;
    mu = 1e-4;
    max_mu = 1e10;
    DEBUG = False;
    
    # Lambda
    dim = X.shape
    if np.isnan(lambda_):
        lambda_=1/np.sqrt(np.max([dim[0],dim[1]])*dim[2])
    
    if hasattr(opts, "tol"):
        tol = opts.tol
    if hasattr(opts, "max_iter"):
        max_iter = opts.max_iter
    if hasattr(opts, "rho"):
        rho = opts.rho
    if hasattr(opts, "mu"):
        mu = opts.mu
    if hasattr(opts, "max_mu"):
        max_mu = opts.max_mu
    if hasattr(opts, "DEBUG"):
        DEBUG = opts.DEBUG
    
    # Initialize L, S and Y
    L = np.zeros((dim))
    S = L
    Y = L
    
    ## ITERATIVE PROCESS
    for itercount in range(0,max_iter):
        Lk = L
        Sk = S		
		
        # update L
        L,tnnL,trank_ = prox_tnn(-S + X - Y/mu, 1/mu)
			
        # update S
        S = prox_l1(-L + X - Y/mu, lambda_/mu)
	
        # Compute residual error
        dY = L + S - X
        chgL = np.max(np.abs(Lk.flatten()-L.flatten()))
        chgS = np.max(np.abs(Sk.flatten()-S.flatten()))
        chg = np.max([chgL, chgS, np.max(np.abs(dY.flatten())) ])
		
        # Debug
        if DEBUG:
            if itercount == 1 or np.mod(itercount, 10) == 0:
                obj = tnnL + lambda_*np.linalg.norm(S[:],ord=1)
                err = np.linalg.norm(dY[:])
                print("iter ", iter, ", mu=", mu, ", obj=", obj, ", err=", err)
		
        # Stop condition
        if chg < tol:
            break;
		
        Y = Y + mu*dY;
        mu = np.min([rho*mu, max_mu]);
	
    obj = tnnL + lambda_*np.linalg.norm(S.flatten().flatten(),ord=1)
    err = np.linalg.norm(dY.flatten().flatten());

    return L.real,S.real,obj,err,itercount






# 3D Order tensor print
#
# version 1.0 - 28/03/2019
#
# Written by Fernando Hermosillo
# 
def tprint(T):
    print('Tensor of size ',T.shape,':')
    for i in range(0,T.shape[2]):
        print(T[:,:,i])
        print(' ');
        
        

# END OF FILE #################################################################