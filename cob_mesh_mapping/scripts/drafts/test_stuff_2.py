#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm

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

def plotEig(mu, C, axis):
    Q,S = qs(C)
    #print Q
    axis.arrow(mu[0,0], mu[1,0], .15*Q[0,0], .15*Q[1,0])
    axis.arrow(mu[0,0], mu[1,0], .05*Q[0,1], .05*Q[1,1])

def mat2quat(M):
    w = 0.5 * sqrt(1. + M[0,0] + M[1,1] + 1.)
    x = 0 #3d: (M[2,1]-M[1,2]) / (4.*w)
    y = 0 #3d: (M[0,2]-M[2,0]) / (4.*w)
    z = (M[1,0]-M[0,1]) / (4.*w)
    return array([w,x,y,z])

def quat2mat(q):
    w,x,y,z = q
    xx = 1. - 2.*(y**2 + z**2)
    xy = 2.*(x*y - w*z)
    yx = 2.*(x*y + w*z)
    yy = 1. - 2.*(x**2 + z**2)
    return mat([[xx, xy], [yx, yy]])

''' interpolation of quaternions '''
def slerp(q1,q2,t):
    omega = arccos( dot(q1/linalg.norm(q1), q2/linalg.norm(q2)) )
    #print "Deg: ",rad2deg(omega)
    so = 1./sin(omega)
    if linalg.norm(q1 - q2) < linalg.norm(q1 + q2):
        q = sin((1.-t)*omega)*so*q1 + sin(t*omega)*so*q2
    else:
        q = sin((1.-t)*omega)*so*q1 + sin(t*omega)*so*-q2
    return q

''' covariance linear interpolation '''
def covLerp(C1,C2,t):
    Q1,S1 = qs(C1)
    Q2,S2 = qs(C2)
    Q = quat2mat( slerp(mat2quat(Q1), mat2quat(Q2), t) )
    S = (1.-t)*S1 + t*S2
    return Q*S*Q.T

def qs(M):
    #print "det:", linalg.det(M)
    U,s,V = linalg.svd(M)
    Q = mat(U)
    if Q[1,0] > 0: Q = -1.*Q
    if linalg.det(Q) < 0:
        Q[:,1] = -1.*Q[:,1]

    S = mat(diag(s))
    return Q,S

''' project point p on line v '''
def projectOnLine(v, p):
    l = 1./linalg.norm(v)
    vn = v*l
    a = float(vn.T*p)
    u = a*vn
    return u,a*l


def affine(p):
    return vstack([p,1.])

def cov2d(v,s):
    return s.covariance(vstack([0,v]))[1:,1:]


sensor = sm.SensorModel()
v1 = mat([-0.8,2.3]).T
v2 = mat([0.4,2.3]).T
v3 = mat([0.1,2.0]).T
#v3 = mat([0.9,1.8]).T
v4 = mat([0.9,1.4]).T

points = array(hstack([v1,v2,v4,v1])).T

V1 = cov2d(v1,sensor)
V2 = cov2d(v2,sensor)
V3 = cov2d(v3,sensor)
V4 = cov2d(v4,sensor)

u1, a1 = projectOnLine(v2-v1,v3-v1)
U1 = covLerp(V1, V2, a1)
u2, a2 = projectOnLine(v4-v1, v3-v1)
U2 = covLerp(V1, V4, a2)
u3, a3 = projectOnLine(v4-v2, v3-v2)
U3 = covLerp(V2, V4, a3)

Y = hstack([affine(v1+u1),affine(v1+u2),affine(v2+u3)])
b = linalg.inv(Y)*affine(v3)
V3f = b[0,0]*U1 + b[1,0]*U2 + b[2,0]*U3

X = hstack([affine(v1),affine(v2),affine(v4)])
a = linalg.inv(X)*affine(v3)
V3e = a[0,0]*V1 + a[1,0]*V2 + a[2,0]*V4

minx = min(points[:,0])
maxx = max(points[:,0])
miny = min(points[:,1])
maxy = max(points[:,1])

xx1 = [minx-.1,maxx+.1]
xx2 = [miny-.15,maxy+.15]


fig =  plt.figure()
fig.clf()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(points[:,0],points[:,1],'-')

plotCov(v1,10.*V1,ax)
plotEig(v1,V1,ax)
plotCov(v2,10.*V2,ax)
plotEig(v2,V2,ax)
plotCov(v4,10.*V4,ax)
plotEig(v4,V4,ax)

switch = 3
if switch==1:
    plotCov(v1+u1,10.*cov2d(v1+u1,sensor),ax)
    plotCov(v1+u2,10.*cov2d(v1+u2,sensor),ax)
    plotCov(v2+u3,10.*cov2d(v2+u3,sensor),ax)
    plotEig(v1+u1,cov2d(v1+u1,sensor),ax)
    plotEig(v1+u2,cov2d(v1+u2,sensor),ax)
    plotEig(v2+u3,cov2d(v2+u3,sensor),ax)

    plotCov(v3,10.*V3,ax)
    plotEig(v3,V3,ax)
elif switch==2:
    plotCov(v3,10.*V3e,ax)
    plotEig(v3,V3e,ax)
else:
    plotCov(v1+u1,10.*U1,ax)
    plotCov(v1+u2,10.*U2,ax)
    plotCov(v2+u3,10.*U3,ax)
    plotEig(v1+u1,U1,ax)
    plotEig(v1+u2,U2,ax)
    plotEig(v2+u3,U3,ax)

    plotCov(v3,10.*V3f,ax)
    plotEig(v3,V3f,ax)


ax.axis('equal')
ax.axis([xx1[0],xx1[1],xx2[0],xx2[1]])
plt.show()

