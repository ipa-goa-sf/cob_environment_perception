#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# mu + randn() * sigma -> N(mu, sigma^2)
def var(y):
    return 0.01*y**2

n = array([-1.,1.2]) 
n = n / linalg.norm(n)
beta_t = array([ n[0], n[1], -0.25])
beta_t = beta_t / beta_t[-1]
nn = 10
#samples = hstack([-ones([nn,1]),zeros([nn,1]),array([linspace(0,2.,nn)]).T])
samples = hstack([-ones([nn,1]),zeros([nn,1]),2*random.rand(nn,1)+0.5])

xx = zeros([nn,3]) # measurements
for i in range(nn):
    xx[i,:] = cross(beta_t,samples[i,:])
    xx[i,:] = xx[i,:] / xx[i,-1]
    xx[i,1] = xx[i,1] + random.randn() * var(xx[i,1])

cov = mat(zeros([3,3]))
#mu2 = mat(zeros([3,1]))
for i in range(nn):
    x = mat(xx[i,:]).T
    sig2_inv = 1./var(xx[i,1])**2
    cov = cov + x * x.T * sig2_inv
    #mu2 = mu2 + x * .5 * sig2_inv

#B1,B2 = meshgrid(linspace(0,5,100),linspace(-5,0,100))
B1,B2 = meshgrid(linspace(-.01,.01,100),linspace(-.01,.01,100))
Y = zeros(shape(B1))
Y2 = zeros(shape(B1))

A = cov/cov[-1,-1]
cov = linalg.inv(cov)
mu2 = cov[:-1,-1] / cov[-1,-1]
cov2 = cov[:-1,:-1] - mu2 * cov[-1,:-1]
cov2_inv = linalg.inv(cov2)

print linalg.inv(A)
print cov
# plot parameter space -> possible betas
for yi in range(shape(B1)[0]):
    for xi in range(shape(B1)[1]):
        b = mat([[B1[yi,xi]],[B2[yi,xi]],[1.0]])
        Y[yi,xi] = exp(-b.T * A * b )
        b = mat([[B1[yi,xi]],[B2[yi,xi]]])
        Y2[yi,xi] = exp(-b.T * cov2_inv * b )


A[-1,:] = 0
A[-1,-1] = 1.
#cov = linalg.inv(cov)
beta = linalg.inv(A) * mat([[0],[0],[1.]])
grad = beta[:-1,:] / linalg.norm(beta[:-1,:])
grad2 = beta / linalg.norm(beta)
#beta = mat([[2.3],[-2.9],[1.]])
X1,X2 = meshgrid(linspace(0,3,100),linspace(0,3,100))
F = zeros(shape(X1))
for yi in range(shape(X1)[0]):
    for xi in range(shape(X1)[1]):
        x = mat([[X1[yi,xi]],[X2[yi,xi]],[1.]])
        F[yi,xi] = x.T*grad2

# compute variance at specific points based on estimator uncertainty:
mus = zeros(50)
confidence1 = zeros([50,2])
confidence2 = zeros([50,2])
samples_new = hstack([-ones([50,1]),zeros([50,1]),array([linspace(0,3.,50)]).T])
for i in range(50):
    x = mat(cross(hstack(array(beta)),samples_new[i,:])).T
    x = x / x[-1]
    mus[i] = x.T * cov * x
    confidence1[i] = array(x + grad * 3.*mus[i] * 100.)[:,0]
    confidence2[i] = array(x - grad * 3.*mus[i] * 100.)[:,0]

#A = x.T.dot(x)
#mu = hstack([linalg.inv(A[0:-1,0:-1]).dot(-A[-1,0:-1]),[1]])
#mu = linalg.inv(A[0:-1,0:-1]).dot(-A[-1,0:-1])
#sig = xx.T.dot(xx)
#print mu, "\n", beta_t
#disp(A/A[-1,-1])
#mu = sig.dot(x.T).dot(y)


fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

#cs1 = ax1.contour(B1,B2,Y, [2500.,5000.,10000.,20000.])
#cs1 = ax1.contour(B1,B2,Y, 20)
cs1 = ax1.contour(B1,B2,Y2, 20)
ax1.clabel(cs1, inline=1, fontsize=10)
ax2.plot(xx[:,0],xx[:,1], 'x')
cs2 = ax2.contour(X1,X2,F)
ax2.clabel(cs2, inline=1, fontsize=10)
ax2.plot(confidence1[:,0],confidence1[:,1])
ax2.plot(confidence2[:,0],confidence2[:,1])

#ax.axis('equal')
#ax.set_xlim(0, 3)
#ax.set_ylim(0, 3)
ax2.grid()
plt.show()


