#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm
from time import time

from geometry import *
from statistics import *

def randomSensorPoint(n=1):
    tan_fov_2 = 0.5*tan(49./180*pi)
    Y = random.rand(n)*2.+1. #y: [1.,3.)
    wmax = Y*tan_fov_2
    X = random.rand(n)*2.*wmax - wmax
    return vstack([X,Y]).T

def m(a,i):
    return mat(a[i]).T

def computeQ(x1,x2):
    C1 = aff(x1)*aff(x1).T
    C2 = aff(x2)*aff(x2).T
    N = .00001*identity(3)
    Cinv = linalg.inv(C1+C2+N)
    b = Cinv[:,-1]/linalg.norm(Cinv[:-1,-1])
    return b*b.T


n = 1000
sensor = sm.SensorModel()
v1 = randomSensorPoint(n)
v2 = randomSensorPoint(n)
v3 = randomSensorPoint(n)
v4 = randomSensorPoint(n)

Sa  = zeros([n,2,2])
Sb  = zeros([n,2,2])
Sc  = zeros([n,2,2])
qa = zeros([n,4])
qb = zeros([n,4])
qc = zeros([n,4])
Pc = zeros([n,2,2])

p  = zeros([n,2]) # collapsed point
Pt = zeros([n,2,2]) # true variance sensor
Rt = zeros([n,2,2]) # true rotation sensor
St = zeros([n,2,2]) # true scale sensor

it = zeros(n)

for i in range(n):
    V1 = sm.cov2d(m(v1,i),sensor)
    V2 = sm.cov2d(m(v2,i),sensor)
    V3 = sm.cov2d(m(v3,i),sensor)
    V4 = sm.cov2d(m(v4,i),sensor)

    Q1 = computeQ(m(v1,i),m(v2,i))
    Q2 = computeQ(m(v2,i),m(v3,i))
    Q3 = computeQ(m(v3,i),m(v4,i))

    Qinv = linalg.inv(Q1 + 2.*Q2 + Q3)
    p[i] = (Qinv[:-1,-1]/Qinv[-1,-1]).T
    Pt[i] = sm.cov2d(m(p,i),sensor)
    Rt[i],St[i] = decomposeCov(Pt[i])

    R1,S1 = decomposeCov(V1)
    R2,S2 = decomposeCov(V2)
    R3,S3 = decomposeCov(V3)
    R4,S4 = decomposeCov(V4)
    q1 = mat2quat(R1)
    q2 = mat2quat(R2)
    q3 = mat2quat(R3)
    q4 = mat2quat(R4)

    aa = barycentricWeights(m(v1,i),m(v2,i),m(v3,i),m(p,i)).T.A[0,:]
    ab = barycentricWeights(m(v2,i),m(v3,i),m(v4,i),m(p,i)).T.A[0,:]

    Sa[i] = aa[0]*S1 + aa[1]*S2 + aa[2]*S3
    Sb[i] = ab[0]*S2 + ab[1]*S3 + ab[2]*S4

    qa[i] = aa[0]*q1 + aa[1]*q2 + aa[2]*q3
    qb[i] = ab[0]*q2 + ab[1]*q3 + ab[2]*q4

    qc[i] = .5*qa[i] + .5*qb[i]
    Sc[i] = .5*Sa[i] + .5*Sb[i]

    Rc = quat2mat(qc[i])
    Pc[i] = Rc*mat(Sc[i])*Rc.T


# error frobenius norm rotation matrix
ef1 = zeros(n)
ef2 = zeros(n)
ef3 = zeros(n)

# error quaternion angle
ea1 = zeros(n)
ea2 = zeros(n)
ea3 = zeros(n)

for i in range(n):
    ef1[i] = linalg.norm(Rt[i]-quat2mat(qa[i]))
    ef2[i] = linalg.norm(Rt[i]-quat2mat(qb[i]))
    ef3[i] = linalg.norm(Rt[i]-quat2mat(qc[i]))

    qt = mat2quat(Rt[i])
    ea1[i] = qangle(qa[i],qt)
    ea2[i] = qangle(qb[i],qt)
    ea3[i] = qangle(qc[i],qt)

print "\nfrobenius norm error to sensor (mean,std):"
print "1st : ",ef1.mean(),ef1.std()
print "2nd : ",ef2.mean(),ef2.std()
print "both: ",ef3.mean(),ef3.std()

print "\nangular error to sensor (mean,std):"
print "1st : ",ea1.mean(),ea1.std()
print "2nd : ",ea2.mean(),ea2.std()
print "both: ",ea3.mean(),ea3.std()


es1 = zeros(n)
es2 = zeros(n)
es3 = zeros(n)
for i in range(n):
    es1[i] = linalg.norm(St[i]-Sa[i])
    es2[i] = linalg.norm(St[i]-Sb[i])
    es3[i] = linalg.norm(St[i]-Sc[i])

print "\nfrobenius norm scale error (mean,std):"
print "1st : ",es1.mean(),es1.std()
print "2nd : ",es2.mean(),es2.std()
print "both: ",es3.mean(),es3.std()


switch = 4
if switch==1:
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.hist([ea3,ea1,ea2],50)
    ax2.hist([ef3,ef1,ef2],50)
    ax3.hist([es3,es1,es2],50)
    plt.show()
elif switch==2:
    #idx = ea1.argsort()[-5:]
    idx = ea3.argsort()[:5]
    t1 = v1[idx]
    t2 = v2[idx]
    t3 = v3[idx]
    t4 = v4[idx]
    xx = vstack([t1[:,0],t2[:,0],t3[:,0],t4[:,0]])
    yy = vstack([t1[:,1],t2[:,1],t3[:,1],t4[:,1]])

    pp = p[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(xx,yy,'x-')
    col = 'bgrcmykw'
    for i in range(shape(pp)[0]):
        ax.plot(pp[i,0],pp[i,1],'o'+col[i%len(col)])
    plt.show()
elif switch==3:
    idx = es3.argsort()[-5:]
    #idx = es.argsort()[:5]
    t1 = v1[idx]
    t2 = v2[idx]
    t3 = v3[idx]
    t4 = v4[idx]
    xx = vstack([t1[:,0],t2[:,0],t3[:,0],t4[:,0]])
    yy = vstack([t1[:,1],t2[:,1],t3[:,1],t4[:,1]])

    pp = p[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(xx,yy,'x-')
    col = 'bgrcmykw'
    for i in range(shape(pp)[0]):
        ax.plot(pp[i,0],pp[i,1],'o'+col[i%len(col)])

    plt.show()
elif switch==4:
    idx = es3.argsort()[:5]
    #idx = es3.argsort()[-5:]
    std_idx = int(n*.683)
    idx = es3.argsort()[std_idx:std_idx+5]
    t1 = v1[idx]
    t2 = v2[idx]
    t3 = v3[idx]
    t4 = v4[idx]
    xx = vstack([t1[:,0],t2[:,0],t3[:,0],t4[:,0]])
    yy = vstack([t1[:,1],t2[:,1],t3[:,1],t4[:,1]])

    pp = p[idx]
    PP1 = Pt[idx]
    PP2 = Pc[idx]
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax1.grid()
    ax1.plot(xx,yy,'x-')
    ax2.grid()
    ax2.plot(xx,yy,'x-')
    col = 'bgrcmykw'
    for i in range(shape(pp)[0]):
        ax1.plot(pp[i,0],pp[i,1],'o'+col[i%len(col)])
        ax2.plot(pp[i,0],pp[i,1],'o'+col[i%len(col)])
        plotCov(mat(pp[i]).T,10.*mat(PP1[i]),ax1)
        plotCov(mat(pp[i]).T,10.*mat(PP2[i]),ax2)

    plt.show()
