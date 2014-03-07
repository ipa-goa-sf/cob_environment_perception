#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import mesh_structure as ms
import mesh_optimization as mo
import CohenSutherlandClipping as csclip
import scanline_rasterization as sl
from tf_utils import *

class IterativeMeshLearner:
    def __init__(self):
        self.data = []
        self.mesh = ms.Mesh()
        self.simpler = mo.Simplifier()
        self.simpler.mesh = self.mesh

    #def initMesh(self, measurement):
    #    v1 = self.mesh.add(measurement.m1[0],measurement.m1[1])
    #    v2 = self.mesh.add(measurement.m2[0],measurement.m2[1])
    #    self.mesh.connect(v1,v2)
    #    self.data.append(measurement)

    '''stores measurement as history entry'''
    def addMeasurements(self, m):
        self.data.extend(m)
        #scan = sl.ScanlineRasterization()
        #scan.addEdge(m.m1,m.m2)
        #for e in self.mesh.E:
            #print e
            #scan.addEdge(e.v1.getPos(),e.v2.getPos())

        # rasterize bounding box
        #lim = m.getBoundingBox([.1,.1])
        #grid = scan.fill(lim, [.05,.05])
        # marching cubes mesh reconstruction

    def extendMesh(self, data, cam):
        # first create virtual sensor at current position
        # and reconstruct measurements based on current map
        # however: remember anchor vertices of map where
        # the refined mesh is going to be hooked up on
        v_hooks = []
        self.simpler.reset() # reset simplifier


        v_hooks = [(100.0, None),(-100.0, None)]

        c = csclip.Clipper()
        scan = sl.ScanlineRasterization()
        tf = cam.tf_to_unit_cube.dot(cam.tf_to_cam)
        for e in self.mesh.E:
            w0 = tf.dot(e.v1.getPosAffine()) # redundant transform
            w1 = tf.dot(e.v2.getPosAffine()) # TODO: do better!
            # backface culling (this is just a quick and dirty workaround
            # better: check face normal with some tolerance)
            if (w1[1] - w0[1]) < -.001: continue

            pass0 = pass1 = False
            if w0[-1] > 0:
                pass0, p00, p01 = c.clip(w0,w1) #(+/+)
                if w1[-1] < 0:
                    pass1, p10,p11 = c.clip(-1.*w0,-1.*w1) #(+/-)
            else:
                pass0, p00, p01 = c.clip(-1.*w0,-1.*w1) #(-/-)
                if w1[-1] > 0:
                    pass1, p10, p11 = c.clip(w0,w1) #(-/+)

            if pass0:
                if p00[-1] != 0: p0 = p00/p00[-1]
                else: p0 = p00
                if p01[-1] != 0: p1 = p01/p01[-1]
                else: p1 = p01

                if w0 is not p00 and w0 is not p01: # w0 got clipped
                    # check for smallest clipped y value and save original v
                    if p0[1] < v_hooks[0][0]: v_hooks[0] = (p0[1], e.v1)
                    if p1[1] < v_hooks[0][0]: v_hooks[0] = (p1[1], e.v1)
                    # check for biggest clipped y value and save original v
                    if p0[1] > v_hooks[1][0]: v_hooks[1] = (p0[1], e.v1)
                    if p1[1] > v_hooks[1][0]: v_hooks[1] = (p1[1], e.v1)

                if w1 is not p00 and w1 is not p01: # w1 got clipped
                    # check for smallest clipped y value and save original v
                    if p0[1] < v_hooks[0][0]: v_hooks[0] = (p0[1], e.v2)
                    if p1[1] < v_hooks[0][0]: v_hooks[0] = (p1[1], e.v2)
                    # check for biggest clipped y value and save original v
                    if p0[1] > v_hooks[1][0]: v_hooks[1] = (p0[1], e.v2)
                    if p1[1] > v_hooks[1][0]: v_hooks[1] = (p1[1], e.v2)

                scan.addEdge(p0,p1)
                e.dirty = True # mark for deletion

            if pass1:
                if p10[-1] != 0: p0 = p10/p10[-1]
                else: p0 = p10
                if p11[-1] != 0: p1 = p11/p11[-1]
                else: p1 = p11

                if w0 is not p10 and w0 is not p11: # w0 got clipped
                    # check for smallest clipped y value and save original v
                    if p0[1] < v_hooks[0][0]: v_hooks[0] = (p0[1], e.v1)
                    if p1[1] < v_hooks[0][0]: v_hooks[0] = (p1[1], e.v1)
                    # check for biggest clipped y value and save original v
                    if p0[1] > v_hooks[1][0]: v_hooks[1] = (p0[1], e.v1)
                    if p1[1] > v_hooks[1][0]: v_hooks[1] = (p1[1], e.v1)

                if w1 is not p10 and w1 is not p11: # w1 got clipped
                    # check for smallest clipped y value and save original v
                    if p0[1] < v_hooks[0][0]: v_hooks[0] = (p0[1], e.v2)
                    if p1[1] < v_hooks[0][0]: v_hooks[0] = (p1[1], e.v2)
                    # check for biggest clipped y value and save original v
                    if p0[1] > v_hooks[1][0]: v_hooks[1] = (p0[1], e.v2)
                    if p1[1] > v_hooks[1][0]: v_hooks[1] = (p1[1], e.v2)

                scan.addEdge(p0,p1)
                e.dirty = True # mark for deletion
        # END: for e in self.mesh.E:

        # minor cleanup bug: a anchor vertex (outside current view) is deleted
        #    if it's not connected to any other edge
        self.mesh.cleanup()
        x,y = scan.contour([-1.,1.,-1.,1.], [2./cam.res,2./cam.res])
        vst = vstack(zip(x,y,ones(len(x))))
        m = vstack(v/v[-1] for v in transform(cam.tf_to_frustum,vst))

        # second: combine real measurement data with virtually generated
        # and insert in existing mesh
        ii = len(data[:,0])
        for i in range(ii):
            if m[i][0] > data[i][0]:
                m[i] = data[i]
            elif m[i][0] >= (cam.f):
                m[i][0] = float('nan')


        #print m
        #ax = plt.figure().add_subplot(111)
        #ax.axis('equal')
        #ax.plot(data[:,0],data[:,1],'xr')
        #ax.plot(m[:,0],m[:,1],'xb')

        #m = transform(cam.tf_to_world,m)

        # minor bugs and issues:
        #  - checks for the right anchor is not correct yet
        #    there exist cases where it won't work
        if v_hooks[0][0]<0.0:
            v1 = v_hooks[0][1]
            i = 0
        else:
            for i in range(ii):
                if not math.isnan(m[i][0]): break
            pt = cam.tf_to_world.dot(m[i])
            v1 = self.mesh.add(pt[0],pt[1])

        for j in range(i+1,ii):
            if math.isnan(m[j][0]): continue

            pt = cam.tf_to_world.dot(m[j])
            v2 = self.mesh.add(pt[0],pt[1])
            if fabs(m[j][0] - m[j-1][0]) < .07*(m[j][0])**2+.1:
                e = self.mesh.connect(v1,v2)
                e.dirty = True # mark as new

            v1 = v2

        if v_hooks[1][0]>0.0:
            v2 = v_hooks[1][1]
            #if fabs(m[j][0] - m[j-1][0]) < 0.1:
            e = self.mesh.connect(v1,v2)
            e.dirty = True # mark as new


    def simplifyMesh(self,eps):
        # simplifies the previously extended mesh with respect to all
        # observed measurement data
        for e in self.mesh.E:
            if not e.dirty: continue
            #if e.v1.isBorder():
            #if e.v2.isBorder():
            for d in self.data:
                p = d.resultedFrom(e)
                e.v1.addPlaneParam(d.nx,d.ny, p)
                e.v2.addPlaneParam(d.nx,d.ny, p)

        for e in self.mesh.E:
            if not e.dirty: continue
            self.simpler.markForUpdate(e)

        self.simpler.simplify(eps)
