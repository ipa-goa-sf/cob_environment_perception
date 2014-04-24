#!/usr/bin/python
from numpy import *
import heapq
from collections import namedtuple
import mesh_structure as ms
import matplotlib.pyplot as plt


class Heap:
    def __init__(self):
        self.h = []

    def push(self, cost, data):
        Item = namedtuple('Item', 'cost, data')
        heapq.heappush(self.h,Item(cost,data))

    def pop(self):
        return heapq.heappop(self.h)

    def size(self):
        return len(self.h)

class Simplifier:
    def __init__(self, mesh = None):
        """vertices of mesh require normal information beforhand"""
        self.heap = Heap()
        if mesh is not None:
            self.init(mesh)

    def init(self, mesh):
        self.mesh = mesh
        for e in mesh.E:
            c = self.computeCost(e)
            self.heap.push(c,e)

    def reset(self):
        for h in self.heap.h:
            h.data.dirty = False

        self.heap.__init__()

    def markForUpdate(self, edge):
        c = self.computeCost(edge)
        self.heap.push(c,edge)

    def computeCost(self, edge):
        # compute weak constrains to place v between v1 and v2
        # this will not be saved for future operations
#        ny,nx = edge.getNormal()
#        ny = - ny
#        d = -(nx*edge.v1.x + ny*edge.v1.y)
#        p1 = array([[nx],[ny],[d]])
#        d = -(nx*edge.v2.x + ny*edge.v2.y)
#        p2 = array([[nx],[ny],[d]])

        w = edge.v1.w + edge.v2.w
        Q = edge.v1.Q + edge.v2.Q
        Qw = Q / w
#        Qw = (Q + p1.dot(p1.T) + p2.dot(p2.T)) / (w+2.)
        q = array(vstack([ Qw[0:2,:], [0,0,1.] ]))
        #A = Q[:2,:2]
        #B = Q[2,:2]
        #C = Q[2,2]
        det = fabs(linalg.det(q))
        if det > 0.0001:
            v = linalg.inv(q).dot(array([[0],[0],[1.]]))
            #v = linalg.inv(A).dot(B)
        else:
            v = array([[0.5 * (edge.v1.x + edge.v2.x)],
                       [0.5 * (edge.v1.y + edge.v2.y)],[1]])

        edge.vnew = ms.Vertex(float(v[0]),float(v[1]),Q,w)
        edge.vnew.flag = False
        #cost = float(v.T.dot(A).dot(v) + 2.*v.T.dot(B) + C)
        #cost = fabs(float(v.T.dot(Qw).dot(v)))
        cost = 0.0001 * fabs(float(v.T.dot(Q).dot(v)))
        #print det, cost

        edge.dirty = False
        return cost

    def simplify(self, eps = 0.1, min_vertices = 3):
        h = self.heap.pop()
        while(h.cost < eps
              and len(self.mesh.V) > min_vertices
              and self.heap.size()>0):
            self.mesh.collapse(h.data)
            h = self.heap.pop()
            while(h.data.dirty):
                c = self.computeCost(h.data)
                self.heap.push(c,h.data)
                h = self.heap.pop()

    def simplify2(self, eps, min_vertices=3):
        h = self.heap.pop()
        while(h.cost < eps
              and len(self.mesh.V) > min_vertices
              and self.heap.size() > 0):
            self.mesh.collapse(h.data)
            h = self.heap.pop()
            while(h.data.dirty):
                #h.data.v1.w = 0
                #h.data.v2.w = 0
                #h.data.v1.Q = zeros([3,3])
                #h.data.v2.Q = zeros([3,3])
                #h.data.updateQuadricsAreaWeighted()

                w = h.data.v1.w + h.data.v2.w
                Q = h.data.v1.Q + h.data.v2.Q
                Qw = Q / w
                q = array(vstack([ Qw[0:2,:], [0,0,1.] ]))
                det = fabs(linalg.det(q))
                if det > 0.00000001:
                    v = linalg.inv(q).dot(array([[0],[0],[1.]]))
                else:
                    v = array([[0.5 * (h.data.v1.x + h.data.v2.x)],
                               [0.5 * (h.data.v1.y + h.data.v2.y)],[1]])

                h.data.vnew = ms.Vertex(float(v[0]),float(v[1]),Q,w)
                h.data.vnew.flag = False
                c = fabs(float(v.T.dot(Qw).dot(v)))
                #c = fabs(float(v.T.dot(Q).dot(v)))
                #print c
                h.data.dirty = False

                self.heap.push(c,h.data)
                h = self.heap.pop()

    def simplifyAndPrint(self, eps, fig):
        h = self.heap.pop()
        fii = 0
        while(h.cost < eps
              and len(self.mesh.V) > 3
              and self.heap.size() > 0):

            fig.init('Mesh Simplification')
            self.mesh.draw(fig.ax1, 've', 'bbb')
            fig.save('img_out/mesh_learner_')
            fii = fii+1

            self.mesh.collapse(h.data)
            h = self.heap.pop()
            while(h.data.dirty):
                w = h.data.v1.w + h.data.v2.w
                Q = h.data.v1.Q + h.data.v2.Q
                Qw = Q / w
                q = array(vstack([ Qw[0:2,:], [0,0,1.] ]))
                det = fabs(linalg.det(q))
                if det > 0.00000001:
                    v = linalg.inv(q).dot(array([[0],[0],[1.]]))
                else:
                    v = array([[0.5 * (h.data.v1.x + h.data.v2.x)],
                               [0.5 * (h.data.v1.y + h.data.v2.y)],[1]])

                h.data.vnew = ms.Vertex(float(v[0]),float(v[1]),Q,w)
                h.data.vnew.flag = False
                c = fabs(float(v.T.dot(Qw).dot(v)))
                #c = fabs(float(v.T.dot(Q).dot(v)))
                h.data.dirty = False

                self.heap.push(c,h.data)
                h = self.heap.pop()



#m = ms.Mesh()
#m.test_load()
#m.test_show()
#s = Simplifier()
#s.init(m)
#s.simplify(10.0,5)
#m.test_show()
#plt.show()
