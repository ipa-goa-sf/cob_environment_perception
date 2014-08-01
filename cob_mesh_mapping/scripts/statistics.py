#!/usr/bin/python

from numpy import *

#------------------------------------------------------------------------------
#   covariance matrix operations
#------------------------------------------------------------------------------
def decomposeCov(M):
    """decomposes covariance matrix M into rotation and scale matrix"""
    U,s,V = linalg.svd(M)
    R = mat(U)
    if R[1,0] > 0: R = -1.*R
    if linalg.det(R) < 0:
        R[:,1] = -1.*R[:,1]

    S = mat(diag(s))
    return R,S

def productGaussian(mu1, C1, mu2, C2):
    """product of two Gaussian distributions N(x|mu1,1/C1), N(y|mu2,1/C2)

    mu = (C1+C2)^-1(C1*mu2+C2*mu1)
    C = C1*(C1+C2)^-1*C2
    """
    denom = linalg.inv(C1+C2)
    mu = denom*(C1*mu2+C2*mu1)
    C = C1*denom*C2
    return mu,C


#------------------------------------------------------------------------------
#   plotting
#------------------------------------------------------------------------------
def plotCov(mu, C, axis):
    """create contour plot at mu with covariance C"""
    xx1 = [mu[0,0]-0.2,mu[0,0]+0.2]
    xx2 = [mu[1,0]-0.2,mu[1,0]+0.2]
    X1,X2 = meshgrid(linspace(xx1[0],xx1[1],100), linspace(xx2[0],xx2[1],100))
    f = zeros(shape(X1))
    Cinv = linalg.inv(C)
    for yi in range(shape(X1)[0]):
        for xi in range(shape(X1)[1]):
            x = mat([X1[yi,xi],X2[yi,xi]]).T - mu
            f[yi,xi] = exp(-0.5*x.T * Cinv * x)

    axis.contour(X1,X2,f,5)

def plotEig(mu, C, axis):
    """show eigen directions (not values) of covariance C at mu"""
    Q,S = decomposeCov(C)
    axis.arrow(mu[0,0], mu[1,0], .15*Q[0,0], .15*Q[1,0])
    axis.arrow(mu[0,0], mu[1,0], .05*Q[0,1], .05*Q[1,1])

