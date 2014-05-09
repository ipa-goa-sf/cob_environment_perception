#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt

def var(y):
    return 0.01*y**2

n = array([-1.,1.2])
n = n / linalg.norm(n)
beta_t = array([ n[0], n[1], -0.25])
beta_t = beta_t / beta_t[-1]
nn = 200
nnn = 20
#samples = hstack([-ones([nn,1]),zeros([nn,1]),array([linspace(0,2.,nn)]).T])
samples = hstack([-ones([nn,1]),zeros([nn,1]),2*random.rand(nn,1)+0.5])
sa = array([-1.,0,0.5])
sb = array([-1.,0,2.5])

xx = zeros([nn,3]) # measurements
x = zeros([nn,3]) # measurements
A = zeros([3,3])
for i in range(nn):
    xx[i,:] = cross(beta_t,samples[i,:])
    xx[i,:] = xx[i,:] / xx[i,-1]
    xx[i,1] = xx[i,1] + random.randn() * var(xx[i,1])
    sig_inv = 1./var(xx[i,1])
    x[i,:] = xx[i,:] * sig_inv
    A = A + array(mat(x[i]).T.dot(mat(x[i])))


#A = x.T.dot(x)
#mu = hstack([linalg.inv(A[0:-1,0:-1]).dot(-A[-1,0:-1]),[1]])
mu = linalg.inv(A[0:-1,0:-1]).dot(-A[-1,0:-1])
sig = xx.T.dot(xx)
#print mu, "\n", beta_t
#disp(A/A[-1,-1])
#mu = sig.dot(x.T).dot(y)

#beta_n = random.multivariate_normal(mu,sig/sig[-1,-1],nnn)
#beta_n = random.multivariate_normal(mu,0.001*A/A[-1,-1],nnn)

beta_n = hstack([random.multivariate_normal(mu,A[0:-1,0:-1]/A[-1,-1],nnn),ones([nnn,1])])
print linalg.norm(beta_n-beta_t)
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx[:,0],xx[:,1],'x')
p = zeros([2,3])
for i in range(nnn):
    p[0] = cross(beta_n[i],sa)
    p[1] = cross(beta_n[i],sb)
    p[0] = p[0] / p[0,-1]
    p[1] = p[1] / p[1,-1]
    ax.plot(p[:,0],p[:,1])

#ax.axis('equal')
#ax.set_xlim(0, 3)
#ax.set_ylim(0, 3)
ax.grid()
plt.show()

'''
