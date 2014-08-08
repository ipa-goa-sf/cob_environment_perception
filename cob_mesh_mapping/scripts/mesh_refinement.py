#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
from collections import defaultdict

import sensor_model as sm
from geometry import *
from statistics import *

def distanceCheck(x1, x2):
    return fabs(x1-x2) < (.04*x2**2 + .01)

class Refiner:
    def __init__(self):
        self.s = sm.SensorModel()

    def insertMeasurements(self, mesh, data, tf, fig=None):
        """
        mesh: mesh structure to extend
        data: original measurement data in homogenious coords (narray)
        tf: transformation matrix sensor->world (mat)
        """

        nd = len(data[:,0])
        R = tf[:-1,:-1] # rotation
        t = tf[:-1,-1] # translation / sensor position

        ref = [None for i in range(nd)] # maps data point to edge (1:1)
        cov = [mat(zeros([2,2])) for i in range(nd)] # stores cov at data point
        m = [mat(zeros(2)).T for i in range(nd)] # stores transformed data point
        clusters = defaultdict(list) # maps edge to data points (1:n)
        loose = [] # stores unassigned data points
        el = [] # stores new/updated edges

        # for debugging:
        cov_estimate = []
        mu_estimate = []
        cov_comb = []
        mu_comb = []

        # analyse data and compare to mesh
        for i in range(nd):
            if math.isnan(data[i,1]): continue
            # compute covariances of data and rotate
            cov[i] = R * self.s.covariance([0,data[i,0],data[i,1]])[1:,1:] * R.T
            m[i] = (tf*mat(data[i]).T)[:-1]

            # compute intersections with existing mesh
            #chi_min = 2.*2.3 # 68.27% confidence
            chi_min = 11.8 # 99.73%
            e_min = None
            d1 = m[i] - t
            for e in mesh.E:
                # missing: free space clearing
                d2 = e.v2.p - e.v1.p
                # backface culling
                #if mat([-d2[1,0],d2[0,0]])*d1 < .001: continue
                a,b = lineLineInters(t,d1,e.v1.p,d2)
                a = float(a)
                b = float(b)
                if b > 0 and b < 1.: # if valid intersection
                    p = a*d1+t # point on mesh
                    Rp = quat2mat(slerp(e.v1.q, e.v2.q, b)) # rot at p
                    Sp = (1.-b)*e.v1.S + b*e.v2.S # scale at p
                    Cp = Rp*Sp*Rp.T # covariance at p
                    mu,C = productGaussian(p,Cp,m[i],cov[i])
                    # chi-square test if new point lies within both
                    # confidence limits (there might be some possibilities
                    # to combining these equations)
                    chi_p = (mu-p).T * linalg.inv(Cp) * (mu-p) # prediction
                    chi_d = (mu-m[i]).T * linalg.inv(cov[i]) * (mu-m[i]) # data
                    chi_test = (m[i]-p).T * linalg.inv(Cp+cov[i]) * (m[i]-p)
                    #print p.T,mu.T,m[i].T
                    #print "point: ",m[i].T, p.T
                    #print "p:",chi_p," d:",chi_d," both:",chi_test
                    #print Cp
                    #print cov[i]
                    #print Cp
                    #print chi_test, p.T, mu.T, m[i].T

                    cov_estimate.append(Cp)
                    mu_estimate.append(p)
                    cov_comb.append(C)
                    mu_comb.append(mu)

                    if chi_d < chi_min:
                        chi_min = chi_d
                        intersection = (b, mu, C)
                        e_min = e

            if e_min is not None:
                # store edge, mu_min and C_min
                clusters[e_min].append( intersection )
                ref[i] = e_min
            else:
                loose.append(i)

        if fig is not None:
            fig.init('Data Covariance')
            for i in range(nd):
                if math.isnan(data[i,1]): continue
                plotCov(m[i], cov[i], fig.ax1)
            fig.save('img_out/covD_')

            fig.init('Estimated Covariance')
            for i in range(len(cov_estimate)):
                plotCov(mu_estimate[i], cov_estimate[i], fig.ax1)
            for v in mesh.V:
                plotCov(v.p, v.cov(), fig.ax1)
            fig.save('img_out/covE_')

            fig.init('Combined Covariance')
            for i in range(len(cov_comb)):
                plotCov(mu_comb[i], cov_comb[i], fig.ax1)
            fig.save('img_out/covG_')

        # create new mesh for loose points:
        v1 = None
        last_i = 0
        # start at loose[1] if loose[0] is first data point
        s = (1 if len(loose)>0 and loose[0] == 0 else 0)
        for i in loose[s:]:
            if last_i == i-1: # single step on data array
                # => data[last_i] and data[i] are valid points,
                # do only distance check
                if distanceCheck(data[last_i,1],data[i,1]):
                    if v1 is None: # create v1 if necessary
                        v1 = mesh.add(m[i-1])
                        RC,v1.S = decomposeCov(cov[i-1])
                        v1.q = mat2quat(RC)
                    #else:
                        #print "v1: ",v1.e1, v1.e2
                    # create v2 and connect
                    v2 = mesh.add(m[i])
                    RC,v2.S = decomposeCov(cov[i])
                    v2.q = mat2quat(RC)

                    e = mesh.connect(v1,v2)
                    e.dirty = True
                    el.append(e)
                    v1 = v2
                else: v1 = None
            # end single step
            else: # large step
                # there was an edge inbetween
                if (ref[i-1] is not None
                    and ref[i-1].v2.isBorder()
                    and not math.isnan(data[i-1,1])
                    and distanceCheck(data[i-1,1],data[i,1])):
                    # connect to existing edge
                    v1 = ref[i-1].v2
                    #print "v1: ",v1.e1, v1.e2
                    # create v2 and connect
                    v2 = mesh.add(m[i])
                    RC,v2.S = decomposeCov(cov[i])
                    v2.q = mat2quat(RC)

                    e = mesh.connect(v1,v2)
                    e.dirty = True
                    el.append(e)
                    v1 = v2
                else: #interupted by nan, do nothing
                    v1 = None
            # end large step
            # in either case check if next point belongs to cluster
            if i<nd-1 and ref[i+1] is not None and ref[i+1].v1.isBorder():
                if (distanceCheck(data[i,1],data[i+1,1])
                    and not math.isnan(data[i+1,1])):
                    # if next point is also valid
                    if v1 is None:
                        # if we didn't just create a vertex, create one
                        v1 = mesh.add(m[i])
                        RC,v1.S = decomposeCov(cov[i])
                        v1.q = mat2quat(RC)
                    #else:
                    #    print "v1: ",v1.e1, v1.e2
                    # connect to existing edge
                    v2 = ref[i+1].v1
                    #print "v2: ",v2.e1, v2.e2
                    e = mesh.connect(v1,v2)
                    e.dirty = True
                    el.append(e)
                    v1 = None
            # end forward check
            last_i = i
        #print clusters
        # update intersected edges
        for e in clusters.keys():
            # get a list of b values of current edge and sort
            #print "Intersections: ",len(clusters[e])
            lb = [i[0] for i in clusters[e]]
            idx = argsort(lb)
            v1 = e.v1
            #print e
            for i in idx:
                v2 = mesh.add(clusters[e][i][1])
                #print v2
                RC,v2.S = decomposeCov(clusters[e][i][2])
                v2.q = mat2quat(RC)

                enew = mesh.connect(v1,v2)
                enew.dirty = True
                el.append(enew)
                v1 = v2
            enew = mesh.connect(v1,e.v2)
            enew.dirty = True
            #print enew
            el.append(enew)
            mesh.E.remove(e)

        vb = []
        for v in mesh.V:
            if v.isBorder():
                vb.append(v)
        for i in range(len(vb)):
            for j in range(i+1,len(vb)):
                if linalg.norm(vb[i].p - vb[j].p) < 0.1:
                    e = mesh.connect(vb[i],vb[j])
                    e.dirty = True
                    el.append(e)
                    break

        for e in el:
            e.beta = e.computeBeta()
        for v in mesh.V:
            if v.flag:
                v.computeQ()

        return el
