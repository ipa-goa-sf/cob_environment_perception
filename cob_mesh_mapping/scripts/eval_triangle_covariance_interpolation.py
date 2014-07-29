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
    Q,S = decomposeCov(C)
    axis.arrow(mu[0,0], mu[1,0], .15*Q[0,0], .15*Q[1,0])
    axis.arrow(mu[0,0], mu[1,0], .05*Q[0,1], .05*Q[1,1])


def cov2d(v,s):
    return s.covariance(vstack([0,v]))[1:,1:]


''' creates affine vector of p '''
def aff(p):
    return vstack([p,1.])

''' convert rotation matrix to quaternion '''
def mat2quat(M):
    w = 0.5 * sqrt(1. + M[0,0] + M[1,1] + 1.)
    x = 0 #3d: (M[2,1]-M[1,2]) / (4.*w)
    y = 0 #3d: (M[0,2]-M[2,0]) / (4.*w)
    z = (M[1,0]-M[0,1]) / (4.*w)
    return array([w,x,y,z])

''' convert quaternion to rotation matrix '''
def quat2mat(q):
    w,x,y,z = q
    xx = 1. - 2.*(y**2 + z**2)
    xy = 2.*(x*y - w*z)
    yx = 2.*(x*y + w*z)
    yy = 1. - 2.*(x**2 + z**2)
    return mat([[xx, xy], [yx, yy]])

def qkonj(q):
    return hstack([q[0], -q[1:]])

def qdot(q0,q1):
    return dot(q0,q1)

def qmul(q0,q1):
    return hstack([ q0[0]*q1[0] - dot(q0[1:],q1[1:]),
                    q0[0]*q1[1:] + q1[0]*q0[1:] + cross(q0[1:],q1[1:]) ])

def qlog(q):
    l = linalg.norm(q)
    return hstack([log(l), q[1:]/linalg.norm(q[1:])*arccos(q[0]/l)])

def qexp(q):
    l = linalg.norm(q[1:])
    #print cos(l)
    #print q#[1:]#/l*sin(l)
    return exp(q[0]) * hstack([cos(l),q[1:]/l*sin(l)])

def qinv(q):
    return qkonj(q)/(linalg.norm(q)**2)

    if linalg.norm(q) == 1.:
        return qkonj(q)
    else:
        print "qinv: inversion of non unit quaternion"
        return zeros(4)

''' Spherical Linear intERPolation of quaternions '''
def slerp(q1,q2,t):
    omega = arccos( dot(q1/linalg.norm(q1), q2/linalg.norm(q2)) )
    so = 1./sin(omega)
    if linalg.norm(q1 - q2) < linalg.norm(q1 + q2):
        q = sin((1.-t)*omega)*so*q1 + sin(t*omega)*so*q2
    else:
        q = sin((1.-t)*omega)*so*q1 + sin(t*omega)*so*-q2
    return q

''' decomposes covariance matrix M into rotation and scale matrix '''
def decomposeCov(M):
    U,s,V = linalg.svd(M)
    Q = mat(U)
    if Q[1,0] > 0: Q = -1.*Q
    if linalg.det(Q) < 0:
        Q[:,1] = -1.*Q[:,1]

    S = mat(diag(s))
    return Q,S

''' computes the barycentric weights of point p on triangle (t1,t2,t3) '''
def barycentricWeights(t1,t2,t3,p):
    X = hstack([aff(t1),aff(t2),aff(t3)])
    return linalg.inv(X)*aff(p)

#########
# Barycentric Linear intERPolation of quaternions
##########
''' iterative generalized quaternion interpolation '''
def blerp1(t1,q1,t2,q2,t3,q3,p):
    m0 = (q1+q2+q3)/3
    for i in range(1,99):
        minv = qinv(m0)
        e = t1*qlog(qmul(minv,q1)) + \
            t2*qlog(qmul(minv,q2)) + \
            t3*qlog(qmul(minv,q3))

        m1 = m0 * qexp(e)
        print m0,m1
        m0 = m1

    return m0


sensor = sm.SensorModel()
p = mat([0.1,2.0]).T
Vtrue = cov2d(p,sensor)

v1 = mat([-0.8,2.3]).T
v2 = mat([0.4,2.3]).T
v3 = mat([0.9,1.4]).T
V1 = cov2d(v1,sensor)
V2 = cov2d(v2,sensor)
V3 = cov2d(v3,sensor)
R1,S1 = decomposeCov(V1)
R2,S2 = decomposeCov(V2)
R3,S3 = decomposeCov(V3)
a = barycentricWeights(v1,v2,v3,p)
q=blerp1(a[0,0],mat2quat(R1),a[1,0],mat2quat(R2),a[2,0],mat2quat(R3),p)
R = quat2mat(q/linalg.norm(q))
S = a[0,0]*S1 + a[1,0]*S2 + a[2,0]*S3

points = array(hstack([v1,v2,v3,v1])).T
fig =  plt.figure()
fig.clf()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(points[:,0],points[:,1],'-')

plotCov(v1,10.*V1,ax)
plotEig(v1,V1,ax)
plotCov(v2,10.*V2,ax)
plotEig(v2,V2,ax)
plotCov(v3,10.*V3,ax)
plotEig(v3,V3,ax)


plotEig(p,R*S*R.T,ax)
plotCov(p,10.*R*S*R.T,ax)
plt.show()
