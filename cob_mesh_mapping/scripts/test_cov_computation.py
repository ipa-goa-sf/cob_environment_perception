#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm

''' project point p on line v '''
def projectOnLine(v, p):
    l = 1./linalg.norm(v)
    vn = v*l
    a = float(vn.T*p)
    u = a*vn
    return u,a*l

''' linear interpolation of covariance matrix C1(v1)->C2(v2)'''
def linIntCov(a, C1, C2):
    return a * C2 + (1.-a) * C1
    #return a**2 * C2 + (1.-a)**2 * C1


def weight(x,s):
    p = mat([[0],[0],[x]])
    return s.covariance(p)[-1,-1]

def computeQ(x1,x2,s):
    # note: weight doesn't change much
    C1 = x1*x1.T#/weight(x1[1],s)
    C2 = x2*x2.T#/weight(x2[1],s)
    N = .00001*identity(3)
    return linalg.inv(C1 + C2 + N)

def affine(p):
    return vstack([p,1.])

def cov2d(v,s):
    return s.covariance(vstack([0,v]))[1:,1:]

def plotFit(beta, xx1, xx2, axis, color=''):
    X1,X2 = meshgrid(linspace(xx1[0],xx1[1],100), linspace(xx2[0],xx2[1],100))
    f = zeros(shape(X1))
    for yi in range(shape(X1)[0]):
        for xi in range(shape(X1)[1]):
            x = mat([X1[yi,xi],X2[yi,xi],1.]).T
            f[yi,xi] = exp(-0.5*x.T * beta * x)

    print X1[f==f.max()],X2[f==f.max()]
    axis.contour(X1,X2,f,7,colors=color)

def plotCov(mu, C, axis, color=''):
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



sensor = sm.SensorModel()

v1 = mat([-0.1,2.3]).T
v2 = mat([0.3,2.3]).T
v3 = mat([0.5,2.1]).T
v4 = mat([0.5,2.0]).T

points = array(hstack([v1,v2,v3,v4])).T

V1 = cov2d(v1,sensor)
V2 = cov2d(v2,sensor)
V3 = cov2d(v3,sensor)
V4 = cov2d(v4,sensor)

minx = min(points[:,0])
maxx = max(points[:,0])
miny = min(points[:,1])
maxy = max(points[:,1])

xx1 = [minx-.3,maxx+.3]
xx2 = [miny-.3,maxy+.3]

# Q for collapsing v2 and v3
Q1 = computeQ(affine(v1),affine(v2),sensor)
Q2 = computeQ(affine(v2),affine(v3),sensor)
Q3 = computeQ(affine(v3),affine(v4),sensor)
b1 = Q1[:,-1]/linalg.norm(Q1[:-1,-1])
b2 = Q2[:,-1]/linalg.norm(Q2[:-1,-1])
b3 = Q3[:,-1]/linalg.norm(Q3[:-1,-1])
B1 = b1*b1.T
B2 = b2*b2.T
B3 = b3*b3.T

#Qinv = linalg.inv(Q1 + 2.*Q2 + Q3)
Qinv = linalg.inv(B1 + 2.*B2 + B3)
#Qinv = linalg.inv(B1 + B3)
w = Qinv[:-1,-1]/Qinv[-1,-1]

# origin at v2:
u1, a1 = projectOnLine(v1-v2, w-v2)
U1 = linIntCov(a1, V2, V1)
# origin at v2:
u2, a2 = projectOnLine(v3-v2, w-v2)
U2 = linIntCov(a2, V2, V3)
# origin at v3:
u3, a3 = projectOnLine(v4-v3, w-v3)
U3 = linIntCov(a3, V3, V4)

d1 = linalg.norm(w-u1)
d2 = linalg.norm(w-u2)
d3 = linalg.norm(w-u3)

ninv = 1./(d1+d2+d3)

W = (1.-d1*ninv)**2 * U1 + (1.-d2*ninv)**2 * U2 + (1.-d3*ninv)**2 * U3
#W = (1.-d1*ninv) * U1 + (1.-d2*ninv) * U2 + (1.-d3*ninv) * U3
Wtrue = cov2d(w,sensor)

fig =  plt.figure()
fig.clf()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(points[:,0],points[:,1],'x-')
ohs = array(hstack([w,v2+u1,v2+u2,v3+u3])).T
ax.plot(ohs[:,0],ohs[:,1],'o')
#plotFit(B1,xx1,xx2,ax,'r')
#plotFit(B2,xx1,xx2,ax,'b')
#plotFit(B3,xx1,xx2,ax,'g')
#plotFit(B1 + 2.*B2 + B3,xx1,xx2,ax,'k')
plotCov(v1,10.*V1,ax)
plotCov(v2,10.*V2,ax)
plotCov(v3,10.*V3,ax)
plotCov(v4,10.*V4,ax)

plotCov(v2+u1,10.*U1,ax)
plotCov(v2+u2,10.*U2,ax)
plotCov(v3+u3,10.*U3,ax)
plotCov(w,10.*W,ax)
#plotCov(w,10.*Wtrue,ax)
ax.axis('equal')
ax.axis([xx1[0],xx1[1],xx2[0],xx2[1]])
plt.show()
