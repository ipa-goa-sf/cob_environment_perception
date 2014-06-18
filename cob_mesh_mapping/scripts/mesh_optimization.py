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
            #c = self.computeCost(e)
            e.dirty = True
            self.heap.push(0,e)

    def reset(self):
        for h in self.heap.h:
            h.data.dirty = False

        self.heap.__init__()

    def markForUpdate(self, edge):
        #c = self.computeCost(edge)
        edge.dirty = True
        self.heap.push(0,edge)


    def simplify2(self, eps, min_vertices=3):
        h = self.heap.pop()
        while(h.cost < eps
              and len(self.mesh.V) > min_vertices
              and self.heap.size() > 0):
            self.mesh.collapse(h.data)
            h = self.heap.pop()
            while(h.data.dirty):
                w = h.data.v1.w + h.data.v2.w
                Q = h.data.v1.Q + h.data.v2.Q
                #Qw = Q / w
                q = array(vstack([ Q[0:2,:], [0,0,1.] ]))
                det = fabs(linalg.det(q))
                if det > 0.00000001:
                    v = linalg.inv(q).dot(array([[0],[0],[1.]]))
                else:
                    v = array([[0.5 * (h.data.v1.x + h.data.v2.x)],
                               [0.5 * (h.data.v1.y + h.data.v2.y)],[1]])

                h.data.vnew = ms.Vertex(float(v[0]),float(v[1]),Q,w)
                #h.data.vnew.flag = False
                #c = fabs(float(v.T.dot(Qw).dot(v)))
                c = fabs(float(v.T.dot(Q).dot(v)))
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
            fig.plotPrecision(h.data.v1.Q + h.data.v2.Q)
            fig.save('img_out/mesh_learner_')
            fii = fii+1

            self.mesh.collapse(h.data)
            h = self.heap.pop()
            while(h.data.dirty):
                w = h.data.v1.w + h.data.v2.w
                Q = h.data.v1.Q + h.data.v2.Q
                #Qw = Q / w
                q = array(vstack([ Q[0:2,:], [0,0,1.] ]))
                det = fabs(linalg.det(q))
                if det > 0.00000001:
                    v = linalg.inv(q).dot(array([[0],[0],[1.]]))
                else:
                    v = array([[0.5 * (h.data.v1.x + h.data.v2.x)],
                               [0.5 * (h.data.v1.y + h.data.v2.y)],[1]])

                h.data.vnew = ms.Vertex(float(v[0]),float(v[1]),Q,w)
                #h.data.vnew.flag = False
                #c = fabs(float(v.T.dot(Qw).dot(v)))
                c = fabs(float(v.T.dot(Q).dot(v)))
                print c
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
