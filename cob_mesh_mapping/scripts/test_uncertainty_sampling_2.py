#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sensor_model as sm

def weight(x,s):
    p = mat([[0],[0],[x]])
    return 100.*s.covariance(p)[-1,-1]

sensor = sm.SensorModel()
nn = 100
beta_true1 = mat([-1.,1.2,-0.25]).T
beta_true1 = beta_true1/linalg.norm(beta_true1)
samples1 = mat(hstack([2.*random.rand(nn,1)+1.,zeros([nn,1]),ones([nn,1])]))

beta_true2 = mat([1.,.7,-1.75]).T
beta_true2 = beta_true2/linalg.norm(beta_true2)
samples2 = mat(hstack([2.*random.rand(nn,1)-1.,zeros([nn,1]),ones([nn,1])]))
for i in range(nn):
    samples1[i,1] = samples1[i]*beta_true1 / -beta_true1[1]
    samples1[i,1] = samples1[i,1] + weight(samples1[i,1],sensor)*random.randn()
    samples2[i,1] = samples2[i]*beta_true2 / -beta_true2[1]
    samples2[i,1] = samples2[i,1] + weight(samples2[i,1],sensor)*random.randn()

A = mat(zeros([3,3]))
sig2 = mat(zeros([nn,1]))
B = mat(zeros([3,3]))

for i in range(nn):
    x = samples1[i].T
    sig2[i] = weight(x[1],sensor)
    A = A + x*x.T / sig2[i]
    x = samples2[i].T
    B = B + x*x.T / weight(x[1],sensor)

A = A + 0.001*identity(3)
B = B + 0.001*identity(3)
Ainv = linalg.inv(A)
Binv = linalg.inv(B)
Cinv = linalg.inv(A+B)
mu = Ainv[:-1,-1] / Ainv[-1,-1] # A_12 / A_22
grad = vstack([mu,mat(1.)])
grad = grad / linalg.norm(grad)

# fill grid in euclidean space (x1,x2)
X1,X2 = meshgrid(linspace(-1,3,80),linspace(0,5,80))
F_grad = zeros(shape(X1))
det1 = zeros(shape(X1))
det2 = zeros(shape(X1))
for yi in range(shape(X1)[0]):
    for xi in range(shape(X1)[1]):
        x = mat([[X1[yi,xi]],[X2[yi,xi]],[1.]])
        F_grad[yi,xi] = x.T*grad
        det1[yi,xi] = ( sqrt( (2.*pi)**3 * linalg.det(linalg.inv(x*x.T + A)))
                       * sqrt( (2.*pi)**3 * linalg.det(linalg.inv(x*x.T + B))) )
        det2[yi,xi] = 0.5*(exp(-0.5*x.T*Ainv*x)/(sqrt((2.*pi)**3*linalg.det(A)))
                           *exp(-0.5*x.T*Binv*x)/(sqrt((2.*pi)**3*linalg.det(B))))*1000.
        #det1[yi,xi] = 0.5*exp(-0.5*x.T*Cinv*x)/sqrt((2.*pi)**3*linalg.det(A+B))*1000.
        #det1[yi,xi] = sqrt( (2.*pi)**3 * linalg.det(linalg.inv(x*x.T+A)) )


nnn = 50
mus = zeros(nnn)
conf_p = zeros([nnn,3])
conf_n = zeros([nnn,3])
xmin = min(samples1[:,0].min(),samples2[:,0].min())
xmax = max(samples1[:,0].max(),samples2[:,0].max())
samples_new = mat(hstack([mat(linspace(xmin,xmax,nnn)).T,zeros([nnn,1]),ones([nnn,1])]))
for i in range(nnn):
    samples_new[i,1] = samples_new[i]*grad / -grad[1]
    x = samples_new[i].T
    mus[i] = x.T * Ainv * x #+ weight(x[1],sensor)
    conf_p[i] = array(x + grad * 3.*mus[i] * 1.)[:,0]
    conf_n[i] = array(x - grad * 3.*mus[i] * 1.)[:,0]

# fill grid in paramter space (a,b):
'''
P1,P2 = meshgrid(linspace(-1,1,50),linspace(-1,1,50))
Pp = zeros(shape(P1))
for qi in range(shape(P1)[0]):
    for pi in range(shape(P1)[1]):
        p = mat([[P1[qi,pi]],[P2[qi,pi]]])
        Pp[qi,pi] = p.T*A*p  / 100000.
'''

'''
phi = linspace(-math.pi, math.pi, 50)
p_norm = zeros(shape(phi))
test = zeros(shape(phi))
for i in range(shape(phi)[0]):
    p = mat([[math.cos(phi[i])],[math.sin(phi[i])]])
    p_norm[i] = p.T*A*p
    test[i] = (A[0,1]+A[1,1])*(sin(2.*phi[i])+1.)
'''


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#cs1 = ax1.contour(X1,X2,F_grad,20)
cs1 = ax1.contour(X1,X2,det1,20)
ax1.clabel(cs1,inline=1,fontsize=10)
ax1.plot(samples1[:,0],samples1[:,1], 'x')
ax1.plot(samples2[:,0],samples2[:,1], 'x')
#ax1.plot(conf_p[:,0],conf_p[:,1],'r')
#ax1.plot(conf_n[:,0],conf_n[:,1],'r')
ax1.grid()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
cs2 = ax2.contour(X1,X2,det2,20)
ax2.clabel(cs2,inline=1,fontsize=10)
ax2.plot(samples1[:,0],samples1[:,1], 'x')
ax2.plot(samples2[:,0],samples2[:,1], 'x')
ax2.grid()


'''
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
cs2 = ax2.contour(P1,P2,Pp,20)
ax2.clabel(cs2,inline=1,fontsize=10)
ax2.grid()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(phi,p_norm)
ax3.plot(phi,test)
ax3.grid()
'''
plt.show()
