#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm

from geometry import *
from statistics import *

def distanceCheck(x1, x2):
    return fabs(x1-x2) < (.04*x2**2 + .01)

class Refiner:
    def __init__(self):
        self.s = sm.SensorModel()

    def insertMeasurements(self, mesh, data, tf):
        #fig.init('Covariance')

        R = tf[:-1,:-1]
        el = []
        ii = len(data[:,0])
        for i in range(ii):
            if not math.isnan(data[i,0]): break

        v1 = mesh.add([data[i,0],data[i,1]])
        C = mat(self.s.covariance([0,data[i,0],data[i,1]])[1:,1:])
        CR,v1.S = decomposeCov(R*C*R.T)
        v1.q = mat2quat(CR)
        v1.p = (tf*aff(v1.p))[:-1]
        #plotCov(v1.p,10.*R*C*R.T,fig.ax1)
        broke = False
        for j in range(i+1,ii):
            if math.isnan(data[j,0]):
                broke = True
                continue
            v2 = mesh.add([data[j,0],data[j,1]])
            C = mat(self.s.covariance([0,data[j,0],data[j,1]])[1:,1:])
            CR,v2.S = decomposeCov(R*C*R.T)
            v2.q = mat2quat(CR)
            v2.p = (tf*aff(v2.p))[:-1]
            #plotCov(v2.p,10.*R*C*R.T,fig.ax1)
            if not broke and distanceCheck(data[j,1],data[j-1,1]):
                el.append(mesh.connect(v1,v2))
                el[-1].dirty = True
            broke = False
            v1 = v2
        #fig.save('img_out/cov_')
        for e in el:
            e.updateQ()
        return el
# measurement - mesh intersection

# retriangulation
