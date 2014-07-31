#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm
from time import time

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

''' project point p on line v '''
def projectOnLine(v, p):
    l = 1./linalg.norm(v)
    vn = v*l
    a = float(vn.T*p)
    u = a*vn
    return u,a*l

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

def qangle(q1,q2):
    if linalg.norm(q1 - q2) < linalg.norm(q1 + q2):
        return arccos( dot(q1/linalg.norm(q1), q2/linalg.norm(q2)) )
    else:
        return arccos( dot(q1/linalg.norm(q1), -q2/linalg.norm(q2)) )

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
def blerp1(t1,q1,t2,q2,t3,q3):
    m0 = (q1+q2+q3)/3
    for i in range(1,50):
        minv = qinv(m0)
        e = t1*qlog(qmul(minv,q1)) + \
            t2*qlog(qmul(minv,q2)) + \
            t3*qlog(qmul(minv,q3))

        m1 = m0 * qexp(e)
        #print m0,m1
        m0 = m1

    return m0

''' quaternion averaging '''
def blerp2(t1,q1,t2,q2,t3,q3):
    M = t1*mat(q1).T*mat(q1) + \
        t2*mat(q2).T*mat(q2) + \
        t3*mat(q3).T*mat(q3)
    U,s,V = linalg.svd(M)
    return U.A[:,0]

''' naive quaternion averaging '''
def blerp3(t1,q1,t2,q2,t3,q3):
    return t1*q1+t2*q2+t3*q3

''' iterative slerp '''
def blerp4(v1,q1,v2,q2,v3,q3,p):
    qq1 = q1
    mq1 = q1
    qq2 = q2
    qq3 = q3
    p1 = v1
    p2 = v2
    p3 = v3
    #P = hstack([p1,p2,p3]).A
    i = 1
    while True:
        u1,a1 = projectOnLine(p2-p1,p-p1)
        u2,a2 = projectOnLine(p3-p2,p-p2)
        u3,a3 = projectOnLine(p1-p3,p-p3)
        if (u1.T*u1 + u2.T*u2 + u3.T*u3) < 0.0001 or i>2:
            break
        qq1 = slerp(mq1,qq2,a1)
        qq2 = slerp(qq2,qq3,a2)
        qq3 = slerp(qq3,mq1,a3)
        #print "q: ",qq1,qq2,qq3
        mq1 = qq1
        p1 = p1 + u1
        p2 = p2 + u2
        p3 = p3 + u3
        i = i+1
        #P = hstack([P,p1,p2,p3])
        #print "p: ",p1.T,p2.T,p3.T

    b1,b2,b3 = barycentricWeights(p1,p2,p3,p).A[:,0]
    return b1*qq1 + b2*qq2 + b3*qq3, i

def test():
    sensor = sm.SensorModel()
    #p = mat([0.1,2.0]).T
    p = mat([1.1,2.0]).T
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
    q1 = mat2quat(R1)
    q2 = mat2quat(R2)
    q3 = mat2quat(R3)

    a1,a2,a3 = barycentricWeights(v1,v2,v3,p).A[:,0]
    S = a1*S1 + a2*S2 + a2*S3

    mq1 = blerp1(a1,q1,a2,q2,a3,q3)
    mR1 = quat2mat(mq1/linalg.norm(mq1))
    mC1 = mR1*S*mR1.T
    print "q1: ",mq1," norm: ",linalg.norm(mq1)

    mq2 = blerp2(a1,q1,a2,q2,a3,q3)
    mR2 = quat2mat(mq2)
    mC2 = mR2*S*mR2.T
    print "q2: ",mq2," norm: ",linalg.norm(mq2)

    mq3 = blerp3(a1,q1,a2,q2,a3,q3)
    mR3 = quat2mat(mq3/linalg.norm(mq3))
    mC3 = mR3*S*mR3.T
    print "q3: ",mq3," norm: ",linalg.norm(mq3)

    mq4,it = blerp4(v1,q1,v2,q2,v3,q3,p)
    mR4 = quat2mat(mq4/linalg.norm(mq4))
    mC4 = mR4*S*mR4.T
    print "q4: ",mq4," norm: ",linalg.norm(mq4), "iter: ",it


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

    switch = 0
    if switch == 1:
        plt.title('General')
        plotEig(p,mC1,ax)
        plotCov(p,10.*mC1,ax)
    elif switch == 2:
        plt.title('Avg')
        plotEig(p,mC2,ax)
        plotCov(p,10.*mC2,ax)
    elif switch == 3:
        plt.title('Naive Avg')
        plotEig(p,mC3,ax)
        plotCov(p,10.*mC3,ax)
    elif switch == 4:
        plt.title('Iterative Slerp')
        plotEig(p,mC4,ax)
        plotCov(p,10.*mC4,ax)
    elif switch == 0:
        plt.title('Sensor')
        plotEig(p,Vtrue,ax)
        plotCov(p,10.*Vtrue,ax)

    plt.show()

def randomSensorPoint(n=1):
    tan_fov_2 = 0.5*tan(49./180*pi)
    Y = random.rand(n)*2.+1. #y: [1.,3.)
    wmax = Y*tan_fov_2
    X = random.rand(n)*2.*wmax - wmax
    return vstack([X,Y]).T

def m(a,i):
    return mat(a[i]).T


n = 10000
sensor = sm.SensorModel()
v1 = randomSensorPoint(n)
v2 = randomSensorPoint(n)
v3 = randomSensorPoint(n)
S  = zeros([n,2,2])
q1 = zeros([n,4])
q2 = zeros([n,4])
q3 = zeros([n,4])
a = zeros([n,3])
inlier = zeros(n)

p  = randomSensorPoint(n)
Pt = zeros([n,2,2]) # true variance sensor
Rt = zeros([n,2,2]) # true rotation sensor
St = zeros([n,2,2]) # true scale sensor
qe1 = zeros([n,4]) # blerp 3 quaternion estimate (naive)
qe2 = zeros([n,4]) # blerp 2 quaternion estimate (attitude)
qe3 = zeros([n,4]) # blerp 4 quaternion estimate (iterative slerp)
it = zeros(n)

for i in range(n):
    V1 = cov2d(m(v1,i),sensor)
    V2 = cov2d(m(v2,i),sensor)
    V3 = cov2d(m(v3,i),sensor)
    Pt[i] = cov2d(m(p,i),sensor)
    R1,S1 = decomposeCov(V1)
    R2,S2 = decomposeCov(V2)
    R3,S3 = decomposeCov(V3)
    Rt[i],St[i] = decomposeCov(Pt[i])
    q1[i] = mat2quat(R1)
    q2[i] = mat2quat(R2)
    q3[i] = mat2quat(R3)
    a[i] = barycentricWeights(m(v1,i),m(v2,i),m(v3,i),m(p,i)).T.A
    S[i] = a[i,0]*S1 + a[i,1]*S2 + a[i,2]*S3
    if a[i].min() > 0 and a[i].max() < 1.:
        inlier[i] = 1


t0 = time()
for i in range(n):
    qe1[i] = blerp3(a[i,0],q1[i], a[i,1],q2[i], a[i,2],q3[i])
    qe1[i] = qe1[i]/linalg.norm(qe1[i])
t1 = time()
for i in range(n):
    qe2[i] = blerp2(a[i,0],q1[i], a[i,1],q2[i], a[i,2],q3[i])
t2 = time()
for i in range(n):
    qe3[i],it[i] = blerp4(m(v1,i),q1[i], m(v2,i),q2[i], m(v3,i),q3[i], m(p,i))
    qe3[i] = qe3[i]/linalg.norm(qe3[i])
t3 = time()

print "\nTimings:"
print t1-t0
print t2-t1
print t3-t2

# error frobenius norm rotation matrix
ef1 = zeros(n)
ef2 = zeros(n)
ef3 = zeros(n)

# error quaternion angle
ea1 = zeros(n)
ea2 = zeros(n)
ea3 = zeros(n)

for i in range(n):
    ef1[i] = linalg.norm(Rt[i]-quat2mat(qe1[i]))
    ef2[i] = linalg.norm(Rt[i]-quat2mat(qe2[i]))
    ef3[i] = linalg.norm(Rt[i]-quat2mat(qe3[i]))

    qt = mat2quat(Rt[i])
    ea1[i] = qangle(qe1[i],qt)
    ea2[i] = qangle(qe2[i],qt)
    ea3[i] = qangle(qe3[i],qt)

print "\nfrobenius norm error to sensor (mean,std):"
print "blerp1: ",ef1.mean(),ef1.std()
print "blerp2: ",ef2.mean(),ef2.std()
print "blerp3: ",ef3.mean(),ef3.std()

print "\nangular error to sensor (mean,std):"
print "blerp1: ",ea1.mean(),ea1.std()
print "blerp2: ",ea2.mean(),ea2.std()
print "blerp3: ",ea3.mean(),ea3.std()

ef21 = zeros(n)
ef23 = zeros(n)
ea21 = zeros(n)
ea23 = zeros(n)
for i in range(n):
    ef21[i] = linalg.norm(quat2mat(qe2[i])-quat2mat(qe1[i]))
    ef23[i] = linalg.norm(quat2mat(qe2[i])-quat2mat(qe3[i]))

    ea21[i] = qangle(qe2[i],qe1[i])
    ea23[i] = qangle(qe2[i],qe3[i])

print "\nfrobenius norm error to blerp2 (mean,std):"
print "blerp1: ",ef21.mean(),ef21.std()
print "blerp3: ",ef23.mean(),ef23.std()

print "\nangular error to blerp2 (mean,std):"
print "blerp1: ",ea21.mean(),ea21.std()
print "blerp3: ",ea23.mean(),ea23.std()

switch = 1
if switch==1:
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax1.grid()
    ax2.grid()
    ax1.hist([ea2,ea1,ea3],50)
    ax2.hist([ef2,ef1,ef3],50)
    plt.show()
elif switch==2:
    #idx = ea1.argsort()[-5:]
    idx = ea1.argsort()[:5]
    t1 = v1[idx]
    t2 = v2[idx]
    t3 = v3[idx]
    xx = vstack([t1[:,0],t2[:,0],t3[:,0],t1[:,0]])
    yy = vstack([t1[:,1],t2[:,1],t3[:,1],t1[:,1]])

    pp = p[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(xx,yy,'x-')
    col = 'bgrcmykw'
    for i in range(shape(pp)[0]):
        ax.plot(pp[i,0],pp[i,1],'o'+col[i%len(col)])
    plt.show()
