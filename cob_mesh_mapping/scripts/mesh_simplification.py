#!/usr/bin/python
from numpy import *
import heapq
from collections import namedtuple
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


class Heap:
    def __init__(self):
        self.h = []

    def push(self, cost, edge):
        Item = namedtuple('Item', 'c, e')
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
        for ho in self.heap.h:
            ho.e.dirty = False
        self.heap.__init__()

    def simplify(self):
        ho = self.heap.pop() #pop first heap object
        while(len(m.V) > 3 and self.heap.size() > 0):
            # while ho is dirty compute cost and push back
            while(ho.e.dirty):
                v,c,Q = e.computeCost()
                ho.e.vnew = Vertex(v)
                ho.e.vnew.Q = Q
                ho.e.dirty = False
                #TODO: compute new quaternion and scale
                self.heap.push(c,ho.e)
                ho = self.heap.pop()

            if(ho.c < .00001): break

    def plotHeapCost(self, ax):
        lines = []
        costs = []
        for ho in self.heap.h:
            lines.append([(ho.e.v1.x(),ho.e.v1.y()), (ho.e.v2.x(),ho.e.v2.y())])
            if ho.e.dirty:
                v,c,Q = ho.e.computeCost()
            else: c = ho.c
            costs.append(c)
        lc = LineCollection( lines, linewidths=(5.), cmap=plt.cm.jet )
        lc.set_array(costs)
        #ax.set_xlim(..)
        #ax.set_ylim(..)
        ax.add_collection(lc)
