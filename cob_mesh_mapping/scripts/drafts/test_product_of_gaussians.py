#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm

from geometry import *
from statistics import *

def computeBeta(x1,x2,lamb=1.):
    # note: weight doesn't change much
    C1 = x1*x1.T#/weight(x1[1],s)
    C2 = x2*x2.T#/weight(x2[1],s)
    N = mat(lamb*identity(3))
    return linalg.inv(C1 + C2 + N)[:,-1]

def weight(S1,S2):
    return 2000.*(S1.trace()+S2.trace())

sensor = sm.SensorModel()

v1 = mat([-0.4,2.3]).T
v2 = mat([0.4,2.3]).T
v3 = mat([0.9,1.8]).T
v4 = mat([0.9,1.4]).T
points = array(hstack([v1,v2,v3,v4])).T

V1 = mat(sm.cov2d(v1,sensor))
V2 = mat(sm.cov2d(v2,sensor))
V3 = mat(sm.cov2d(v3,sensor))
V4 = mat(sm.cov2d(v4,sensor))
R1,S1 = decomposeCov(V1)
R2,S2 = decomposeCov(V2)
R3,S3 = decomposeCov(V3)
R4,S4 = decomposeCov(V4)

mu = zeros([10,2])
for i in range(10):
    b1 = computeBeta(aff(v1),aff(v2),10.**-i)
    b2 = computeBeta(aff(v2),aff(v3),10.**-i)
    b3 = computeBeta(aff(v3),aff(v4),10.**-i)
    Qinv = linalg.inv(b1*b1.T + 2.*b2*b2.T + b3*b3.T)
    mu[i] = (Qinv[:-1,-1]/Qinv[-1,-1]).T.A

b1 = computeBeta(aff(v1),aff(v2),.0001)
b2 = computeBeta(aff(v2),aff(v3),.0001)
b3 = computeBeta(aff(v3),aff(v4),.0001)
b1 = b1/linalg.norm(b1[:-1]) # normalize in x,y
b2 = b2/linalg.norm(b2[:-1]) # normalize in x,y
b3 = b3/linalg.norm(b3[:-1]) # normalize in x,y
w1 = weight(S1,S2)
w2 = weight(S2,S3)
w3 = weight(S3,S4)
cost = zeros(10)
for i in range(10):
    a = (10.-i)/10.
    print a
    b1 = (1.-a)*b1/w1 + a*b1
    b2 = (1.-a)*b2/w2 + a*b2
    b3 = (1.-a)*b3/w3 + a*b3
    Q = b1*b1.T + 2.*b2*b2.T + b3*b3.T
    Qinv = linalg.inv(Q)
    mu[i] = (Qinv[:-1,-1]/Qinv[-1,-1]).T
    cost[i] = aff(mat(mu[i]).T).T*Q*aff(mat(mu[i]).T)




#plt.close('all')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.grid()
ax1.plot(points[:,0],points[:,1],'x-')
plotCov(v1,10.*V1,ax1)
plotCov(v2,10.*V2,ax1)
plotCov(v3,10.*V3,ax1)
plotCov(v4,10.*V4,ax1)
ax1.plot(mu[:,0],mu[:,1],'xk')
ax1.axis('equal')
plt.show()
