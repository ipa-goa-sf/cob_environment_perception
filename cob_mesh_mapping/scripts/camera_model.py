#!/usr/bin/python

from numpy import *
import matplotlib.patches as mpatches
from tf_utils import *

def getPoseVectorField(objects):
    vf = array([[s.pos[0],s.pos[1],s.ori[0],s.ori[1]] for s in objects])
    return vf

def drawPoses(objects,axis):
    vf = array([[s.pos[0],s.pos[1],s.ori[0],s.ori[1]] for s in objects])
    axis.quiver(vf[:,0],vf[:,1],vf[:,2],vf[:,3],facecolor='r',
                width=0.002,headlength=8,headwidth=6,headaxislength=8)

'''provides a virtual camera and its transformations in 2d space'''
class Camera2d:
    """ res: number of pixels (multiple of 2)"""
    def __init__(self, fov, far, near, res = 64.):
        tan_fov_2 = tan(fov*0.5)
        self.res = res
        self.fov = fov
        self.f = f = far
        self.n = n = near
        self.r = r = n * tan_fov_2
        self.l = l = -r
        self.frustum = array([[l,n,1], [-f*tan_fov_2,f,1],
                              [f*tan_fov_2,f,1], [r,n,1]])

        # perspectiv projection matrix:
        depth = f - n # f > n
        width = r - l # r > l
        # scale rectangle to unit cube
        us = mat([[2./width,0,0],
                  [0,2./depth,0],
                  [0,0,1.]])
        # translate rectangle center to origin
        ut = mat([[1,0,-(l+width/2.)],
                  [0,1,-(n+depth/2.)],
                  [0,0,1.]])
        # translate back to position
        txn1 = mat([[1.,0,0],
                    [0,1.,n],
                    [0,0,1.]])
        # scale to original size
        sxfn = mat([[1.,0,0],
                    [0,f/n,0],
                    [0,0,1.]])
        # project frustum to rectangle
        pxn = mat([ [1.,0,0],
                    [0,1.,0],
                    [0,1./n,1.]])
        # translate frustum to origin
        txn2 = mat([[1.,0,0],
                    [0,1.,-n],
                    [0,0,1.]])

        self.tf_to_unit_cube = us*ut*txn1*sxfn*pxn*txn2
        self.tf_to_frustum = linalg.inv(self.tf_to_unit_cube)

    def setPose(self, position, orientation):
        self.pos = mat(position).T
        self.ori = mat(orientation).T / linalg.norm(orientation)
        x = self.ori[1,0]
        y = -self.ori[0,0]
        # rotation and translation matrix:
        #phi = math.atan2(orientation[1], orientation[0])
        #sign = math.copysign(1,phi)
        #phi = fabs(phi)
        #self.tf_to_world = mat([[cos(phi), -sign * sin(phi), position[0]],
        #                        [sign * sin(phi), cos(phi), position[1]],
        #                        [0, 0, 1.]])
        self.tf_to_world = mat([[x,-y,self.pos[0]],
                                [y,x,self.pos[1]],
                                [0,0,1.]])
        self.tf_to_cam = linalg.inv(self.tf_to_world)

    ''' draw field of view to figure '''
    def drawFrustum(self, axis):
        vfrustum = transform(self.tf_to_world, self.frustum)[:,0:2]
        poly = mpatches.Polygon(vfrustum, alpha=.2, fc=(0,.75,0))
        axis.add_patch(poly)

    ''' draws an 'o' at the camera position '''
    def drawPosition(self,axis):
        axis.plot(self.pos[0],self.pos[1], 'o',
                  markeredgecolor=(0,0,0), markerfacecolor=(0,0,0),
                  markersize=2)
