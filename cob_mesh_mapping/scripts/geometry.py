#!/usr/bin/python

from numpy import *

###############################################################################
### Provides a set of functions related to geometrical computation tasks
### NOTE: points are assumed to be numpy matrices and won't work with arrays
###############################################################################

#------------------------------------------------------------------------------
#   conversion function
#------------------------------------------------------------------------------
def aff(p):
    """creates affine vector of p"""
    return vstack([p,1.])

def mat2quat(M):
    """converts rotation matrix to quaternion"""
    w = 0.5 * sqrt(1. + M[0,0] + M[1,1] + 1.)
    x = 0 #3d: (M[2,1]-M[1,2]) / (4.*w)
    y = 0 #3d: (M[0,2]-M[2,0]) / (4.*w)
    z = (M[1,0]-M[0,1]) / (4.*w)
    return array([w,x,y,z])

def quat2mat(q):
    """converts quaternion to rotation matrix"""
    w,x,y,z = q
    xx = 1. - 2.*(y**2 + z**2)
    xy = 2.*(x*y - w*z)
    yx = 2.*(x*y + w*z)
    yy = 1. - 2.*(x**2 + z**2)
    return mat([[xx, xy], [yx, yy]])

#------------------------------------------------------------------------------
#   geometrical computation
#------------------------------------------------------------------------------
def projectOnLine(v, p):
    """project point p on line v"""
    l = 1./linalg.norm(v)
    vn = v*l
    a = float(vn.T*p)
    u = a*vn
    return u,a*l

def lineLineInters(p1,d1,p2,d2):
    """computes line-line intersection as a*d1 + p1 = b*d2 + p2

    p1 -- point on line 1
    d1 -- direction of line 1 (not normilzed)
    p2 -- point on line 2
    d1 -- direction of line 2 (not normilzed)

    return mat([a,b]).T
    """
    return linalg.inv(hstack([d1,-d2]))*(p2-p1)

def barycentricWeights(t1,t2,t3,p):
    """computes the barycentric weights of point p on triangle (t1,t2,t3)"""
    X = hstack([aff(t1),aff(t2),aff(t3)])
    return linalg.inv(X)*aff(p)

#------------------------------------------------------------------------------
#   Quaternion functions
#      (not sure if these are fully correct)
#------------------------------------------------------------------------------
def qconj(q):
    """compute conjugate of quaternion q"""
    return hstack([q[0], -q[1:]])

def qdot(q0,q1):
    """compute dot product of quaternion q0 and q1"""
    return dot(q0,q1)

def qmul(q0,q1):
    """compute product of quaternion q0 and q1"""
    return hstack([ q0[0]*q1[0] - dot(q0[1:],q1[1:]),
                    q0[0]*q1[1:] + q1[0]*q0[1:] + cross(q0[1:],q1[1:]) ])

def qlog(q):
    """compute natural logarithm of quaternion q"""
    l = linalg.norm(q)
    return hstack([log(l), q[1:]/linalg.norm(q[1:])*arccos(q[0]/l)])

def qexp(q):
    """compute exponatial of quaternion q"""
    l = linalg.norm(q[1:])
    return exp(q[0]) * hstack([cos(l),q[1:]/l*sin(l)])

def qinv(q):
    """compute inverse of quaternion q"""
    return qkonj(q)/(linalg.norm(q)**2)

def qangle(q1,q2):
    """compute (smaller) angle between quaternion q1 and q2"""
    if linalg.norm(q1 - q2) < linalg.norm(q1 + q2):
        return arccos( dot(q1/linalg.norm(q1), q2/linalg.norm(q2)) )
    else:
        return arccos( dot(q1/linalg.norm(q1), -q2/linalg.norm(q2)) )

def slerp(q0,q1,t):
    """spherical linear interpolation of quaternions q0 and q1 at t"""
    omega = arccos( dot(q0/linalg.norm(q0), q1/linalg.norm(q1)) )
    so = 1./sin(omega)
    if linalg.norm(q0 - q1) < linalg.norm(q0 + q1):
        q = sin((1.-t)*omega)*so*q0 + sin(t*omega)*so*q1
    else:
        q = sin((1.-t)*omega)*so*q0 + sin(t*omega)*so*-q1
    return q
