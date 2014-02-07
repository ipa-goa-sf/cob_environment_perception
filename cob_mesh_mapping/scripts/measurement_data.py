#!/usr/bin/python

from numpy import *

from camera_model import *
import mesh_structure
import CohenSutherlandClipping as csclip

def convertMeshToMeasurementData(mesh, cam):
    res = []
    for e in mesh.E:
        #print e.v1.x, e.v1.y, " | ", e.v2.x, e.v2.y
        if math.isnan(e.v1.x) or math.isnan(e.v2.x): continue
        m1 = cam.tf_to_world.dot(array([e.v1.x, e.v1.y, 1.0]))
        m2 = cam.tf_to_world.dot(array([e.v2.x, e.v2.y, 1.0]))
        res[len(res):] = [MeasurementData(cam.pos,cam.ori,m1,m2)]
    return res

class MeasurementData(Camera2d):
    '''
    Measurement is counter-clock-wise:\n
    p: sensor position
    o: sensor orientation
    m1: measurement point 1
    m2: measurement point 2
    '''
    def __init__(self, p, o, m1, m2): #orientiation is wrong here!
        self.m1 = m1
        self.m2 = m2
        lm1 = linalg.norm(m1)
        lm2 = linalg.norm(m2)
        fov = math.acos(m2.T.dot(m1)/(lm1*lm2))
        f = math.cos(0.5*fov)*max(lm1,lm2)
        Camera2d.__init__(self, fov, f, 0.4)
        Camera2d.setPose(self, p, o)
        # todo: save plane param nx,ny

    '''compute propability that this measurement resulted from a given edge'''
    def resultedFrom(self, edge):
        w1 = self.tf_to_cam.dot(edge.v1.getPosAffine())
        w2 = self.tf_to_cam.dot(edge.v2.getPosAffine())
        if (w2[1] - w1[1]) < -.001: return 0.0
        # do line intersection for p->m1, p->m2, p->w1, p->w2
        return 1.0

    '''padding: space between bounding box and measurement'''
    def getBoundingBox(self, padding = [0,0] ):
        xmin = min(self.m1[0],self.m2[0])
        xmax = max(self.m1[0],self.m2[0])
        ymin = min(self.m1[1],self.m2[1])
        ymax = max(self.m1[1],self.m2[1])
        return [xmin - padding[0],
                xmax + padding[0],
                ymin - padding[1],
                ymax + padding[1]]

    def draw(self, axis):
        axis.plot([self.m1[0], self.m2[0]], [self.m1[1], self.m2[1]], 'bx-')

    def drawBoundingBox(self, axis, padding = [0,0]):
        bb = self.getBoundingBox(padding)
        x = [bb[0], bb[0], bb[1], bb[1], bb[0]]
        y = [bb[2], bb[3], bb[3], bb[2], bb[2]]
        axis.plot(x,y,'r')
