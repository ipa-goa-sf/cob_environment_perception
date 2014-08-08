#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt

from geometry import *

class Vertex:
    def __init__(self, p, q=zeros(4), S=identity(2), Q=zeros([3,3])):
        """p: position, q: quaternion, S: scale, Q: quadric"""
        if shape(p) != (2,1):
            self.p = mat([float(p[0]),float(p[1])]).T
        else:
            self.p = mat(p)
        self.q = array(q)
        self.S = mat(S)
        self.Q = mat(Q)
        self.e1 = None
        self.e2 = None
        self.flag = True

    def x(self): return self.p[0,0]
    def y(self): return self.p[1,0]

    def cov(self):
        R = quat2mat(self.q)
        return R*self.S*R.T

    def computeQ(self):
        if self.e1 is not None:
            self.Q = self.e1.beta * self.e1.beta.T
        else:
            self.Q = 100.*identity(3)
        if self.e2 is not None:
            self.Q = self.Q + self.e2.beta*self.e2.beta.T
        else:
            self.Q = self.Q + 100.*identity(3)
        self.flag = False

    def isDead(self):
        return ( self.e1 is None and self.e2 is None )

    def isBorder(self):
        return ( self.e1 is None or self.e2 is None )

    def __repr__(self):
        return str(id(self))+": v(%3.2f %3.2f)" % (self.p[0,0], self.p[1,0])

    def __hash__(self): return id(self)
    def __eq__(self,other): return id(self) == id(other)

class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.vnew = v1
        self.dirty = False
        self.beta = self.computeBeta()

    def computeBeta(self):
        """computes paramter for line distance function x.T*(b*b.T)*x"""
        C1 = aff(self.v1.p)*aff(self.v1.p).T
        C2 = aff(self.v2.p)*aff(self.v2.p).T
        N = .00001*identity(3)
        Cinv = linalg.inv(C1 + C2 + N)
        self.v1.flag = True
        self.v2.flag = True
        beta = Cinv[:,-1]/linalg.norm(Cinv[:-1,-1]) # normalize in x,y
        w = 2000.*(self.v1.S.trace() + self.v2.S.trace())
        #print w
        a = 1.
        beta = (1.-a)*beta/w + a*beta
        #print beta.T
        return beta
    '''
    def updateQ(self):
        Q = self.computeQ()
        self.v1.Q = self.v1.Q + Q
        self.v2.Q = self.v2.Q + Q

        if self.v1.isBorder():
            self.v1.Q = self.v1.Q + 100.*identity(3)
        if self.v2.isBorder():
            self.v2.Q = self.v2.Q + 100.*identity(3)
    '''
    def computeCost(self):
        Q = self.v1.Q + self.v2.Q
        Qinv = linalg.inv(Q)
        v = Qinv[:-1,-1]/Qinv[-1,-1]
        c = float(aff(v).T*Q*aff(v))
        return v,c,Q


    def __repr__(self):
        return str(id(self))+`self.v1.__repr__()` +"  <--->  "+ `self.v2.__repr__()`


    def __hash__(self): return id(self)
    def __eq__(self,other): return id(self) == id(other)

class Mesh:
    def __init__(self):
        """ """
        self.V = []
        self.E = []

    def add(self,p):
        """ """
        v = Vertex(p)
        self.V.append(v)
        return v

    def connect(self,v1,v2):
        """ """
        e = Edge(v1,v2)
        #if v1.e1 is None: v1.e1 = e
        #else: v1.e2 = e
        #if v2.e1 is None: v2.e1 = e
        #else: v2.e2 = e
        v1.e2 = e
        v2.e1 = e
        self.E.append(e)
        return e

    def collapse(self, e):
        """performs edge collapse operation"""
        vnew = e.vnew

        p2 = e.v1.p; p3 = e.v2.p
        q1 = zeros(4);     q2 = e.v1.q; q3 = e.v2.q; q4 = zeros(4)
        S1 = zeros([2,2]); S2 = e.v1.S; S3 = e.v2.S; S4 = zeros([2,2])

        a = zeros(4)
        if e.v1.e1 is not None:
            p1 = e.v1.e1.v1.p
            q1 = e.v1.e1.v1.q
            S1 = e.v1.e1.v1.S
            a1 = barycentricWeights(p1,p2,p3,vnew.p)
            a[:3] = a[:3] + a1
            vnew.e1 = e.v1.e1
            vnew.e1.v2 = vnew
            vnew.e1.dirty = True
        if e.v2.e2 is not None:
            p4 = e.v2.e2.v2.p
            q4 = e.v2.e2.v2.q
            S4 = e.v2.e2.v2.S
            a2 = barycentricWeights(p2,p3,p4,vnew.p)
            a[1:] = a[1:] + a2
            vnew.e2 = e.v2.e2
            vnew.e2.v1 = vnew
            vnew.e2.dirty = True
        q = a[0]*q1 + a[1]*q2 + a[2]*q3 + a[3]*q3
        S = a[0]*S1 + a[1]*S2 + a[2]*S3 + a[3]*S3
        if a[0]!=0 and a[-1]!=0:
            q = 0.5*q
            S = 0.5*S

        if S[0,0] < 0 or S[1,1] < 0:
            print "Trace:",S.trace()
            print "Sum a1:",a1.sum()
            print "Sum a2:",a1.sum()
            print "a1:",a1.T
            print "a2:",a2.T
            print "p1:",p1.T
            print "p2:",p2.T
            print "p3:",p3.T
            print "p4:",p4.T
            print "p5:",vnew.p.T
            print "l1:",linalg.norm(p1-p2)
            print "l2:",linalg.norm(p2-p3)
            print "l3:",linalg.norm(p3-p4)
            print "S1:\n",S1
            print "S2:\n",S2
            print "S3:\n",S3
            print "S4:\n",S4
            print "S:\n",S

        vnew.q = q/linalg.norm(q)
        vnew.S = S


        #print "\nV: ",len(self.V)
        #for vi in self.V: print vi
        #print "\nE: ",len(self.E)
        #for ei in self.E: print ei
        print "insert: ",vnew
        print "delete: ",e
        self.V.append(vnew)
        self.V.remove(e.v1)
        self.V.remove(e.v2)
        self.E.remove(e)


    def cleanup(self):
        """removes all edges marked as dirty and
        vertices with no edge assigned"""
        tmp_e = []
        tmp_v = []
        for e in self.E:
            if e.dirty:
                if e.v1.e1 is e: e.v1.e1 = None
                else:            e.v1.e2 = None

                if e.v2.e1 is e: e.v2.e1 = None
                else:            e.v2.e2 = None
            else:
                tmp_e.append(e)

        for v in self.V:
            if not v.isDead():
                tmp_v.append(v)

        self.V = tmp_v
        self.E = tmp_e

    def getPoints(self):
        return hstack([v.p for v in self.V]).T.A

    def draw(self, axis, options = 've', color = 'kbr'):
        """ options: e=edges, v=vertices """
        if 'v' in options and len(self.V)>0:
            P = self.getPoints()
            axis.plot(P[:,0],P[:,1],'x'+color[0])
            for v in self.V:
                if v.isBorder():
                    axis.plot(v.x(),v.y(),'xm')

        if 'e' in options and len(self.E)>0:
            for e in self.E:
                x = [e.v1.x(), e.v2.x()]
                y = [e.v1.y(), e.v2.y()]
                axis.plot(x,y,color[1])
