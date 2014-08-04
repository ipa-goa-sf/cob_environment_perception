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

    def insertMeasurements(self, mesh, coords):
        el = []
        ii = len(coords[:,0])
        for i in range(ii):
            if not math.isnan(coords[i,0]): break

        v1 = mesh.add(coords[i])
        R,v1.S = decomposeCov(sm.cov2d(v1.p,self.s))
        v1.q = mat2quat(R)
        broke = False
        for j in range(i+1,ii):
            if math.isnan(coords[j,0]):
                broke = True
                continue
            v2 = mesh.add(coords[j])
            R,v2.S = decomposeCov(sm.cov2d(v2.p,self.s))
            v2.q = mat2quat(R)
            if not broke and distanceCheck(coords[j,0],coords[j-1,0]):
                el.append(mesh.connect(v1,v2))
                el[-1].dirty = True
            broke = False
            v1 = v2

        for e in el:
            e.updateQ()
        return el
# measurement - mesh intersection

# retriangulation
