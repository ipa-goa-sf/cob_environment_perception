#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm

''' convert rotation matrix to quaternion '''
def m2q(M):
    w = 0.5 * sqrt(1. + M[0,0] + M[1,1] + 1.)
    x = 0 #3d: (M[2,1]-M[1,2]) / (4.*w)
    y = 0 #3d: (M[0,2]-M[2,0]) / (4.*w)
    z = (M[1,0]-M[0,1]) / (4.*w)
    return array([w,x,y,z])

def q2m(q):
    w,x,y,z = q
    xx = 1. - 2.*(y**2 + z**2)
    xy = 2.*(x*y + w*z)
    yy = 1. - 2.*(x**2 + z**2)
    #xz = 2.*(x*z + w*y)
    #yz = 2.*(y*z + w*x)
    #zz = 1. - 2.*(x**2 + y**2)
    return mat([[xx, xy], [xy, yy]])

''' interpolation of quaternions '''
def slerp(q1,q2,t):
    omega = arccos( dot(q1/linalg.norm(q1), q2/linalg.norm(q2)) )
    so = 1./sin(omega)
    q = sin((1.-t)*omega)*so*q1 + sin(t*omega)*so*q2
    return q

''' polar decomposition M = QS '''
def qs(M):
    # 2d only, 3d requires iterative approach (see paper)
    #Mt = mat([[M[1,1],-M[1,0]], [-M[0,1],M[0,0]]])
    #Q = M + sign(linalg.det(M)) * Mt
    #S = linalg.inv(Q)*M
    U,s,V = linalg.svd(M)
    Q = mat(U)
    Q[:,1] = -1.*Q[:,1]
    S = Q*mat(diag(s))*Q.T
    return Q,S

''' covariance linear interpolation '''
def covLerp(C1,C2,t):
    Q1,S1 = qs(C1)
    Q2,S2 = qs(C2)
    Q = q2m( slerp(m2q(Q1), m2q(Q2), t) )
    print Q
    S = (1.-t)*S1 + t*S2
    print S
    A = Q*S
    print A
    return A*C1*A.T


''' project point p on line v '''
def projectOnLine(v, p):
    l = 1./linalg.norm(v)
    vn = v*l
    a = float(vn.T*p)
    u = a*vn
    return u,a*l

''' linear interpolation of covariance matrix C1(v1)->C2(v2)'''
def linIntCov(a, C1, C2):
    return (a) * C2 + (1.-a) * C1
    #return a**2 * C2 + (1.-a)**2 * C1

    #return a**2 * C2 + (1.-a**2) * C1


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

v1 = mat([-0.8,2.3]).T
v2 = mat([0.4,2.3]).T
v3 = mat([0.9,1.8]).T
v4 = mat([0.9,1.4]).T

points = array(hstack([v1,v2,v3,v4])).T

V1 = cov2d(v1,sensor)
V2 = cov2d(v2,sensor)
V3 = cov2d(v3,sensor)
V4 = cov2d(v4,sensor)

U,S,V = linalg.svd(V1)
T1 = mat(U)
T1[:,0] = T1[:,0] * S[0]
T1[:,1] = T1[:,1] * S[1]

U,S,V = linalg.svd(V2)
T2 = mat(U)
T2[:,0] = T2[:,0] * S[0]
T2[:,1] = T2[:,1] * S[1]

A = T2*linalg.inv(T1)

minx = min(points[:,0])
maxx = max(points[:,0])
miny = min(points[:,1])
maxy = max(points[:,1])

xx1 = [minx-.1,maxx+.1]
xx2 = [miny-.15,maxy+.15]

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
#u1, a1 = projectOnLine(v1-v2, w-v2)
#U1 = linIntCov(a1, V2, V1)
u1, a1 = projectOnLine(v2-v1,w-v1)
U1 = linIntCov(a1, V1, V2)
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
ohs = array(hstack([w,v1+u1,v2+u2,v3+u3])).T
ax.plot(ohs[:,0],ohs[:,1],'o')
#plotFit(B1,xx1,xx2,ax,'r')
#plotFit(B2,xx1,xx2,ax,'b')
#plotFit(B3,xx1,xx2,ax,'g')
#plotFit(B1 + 2.*B2 + B3,xx1,xx2,ax,'k')
plotCov(v1,10.*V1,ax)
#plotCov(v2,10.*V2,ax)
#plotCov(v3,10.*V3,ax)
#plotCov(v4,10.*V4,ax)

plotCov(.5*(v1+v2),10*covLerp(V1,V2,.5),ax)
plotCov(v2,10*covLerp(V1,V2,.1),ax)

#plotCov(v1+u1,10.*U1,ax)
#plotCov(v2+u2,10.*U2,ax)
#plotCov(v3+u3,10.*U3,ax)

#plotCov(v2+u1,10.*cov2d(v2+u1,sensor),ax)
#plotCov(v2+u2,10.*cov2d(v2+u2,sensor),ax)
#plotCov(v3+u3,10.*cov2d(v3+u3,sensor),ax)

#plotCov(w,10.*W,ax)
#plotCov(w,10.*Wtrue,ax)
ax.axis('equal')
ax.axis([xx1[0],xx1[1],xx2[0],xx2[1]])
plt.show()

