#!/usr/bin/python
from numpy import *
import heapq
from collections import namedtuple
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import mesh_structure as ms
from statistics import *


class Heap:
    def __init__(self):
        self.h = []

    def push(self, cost, edge):
        Item = namedtuple('Item', 'c, e')
        heapq.heappush(self.h,Item(cost,edge))

    def pop(self):
        return heapq.heappop(self.h)

    def size(self):
        return len(self.h)

class Simplifier:
    def __init__(self):
        self.heap = Heap()

    def initHeap(self, edges):
        self.heap = Heap()
        for e in edges:
            e.dirty = True
            self.heap.push(0,e)

    def simplify(self,mesh,edges=[],fig=None):
        if len(edges)==0:
            self.initHeap(mesh.E)
        else:
            self.initHeap(edges)

        ho = self.heap.pop() #pop first heap object
        while(len(mesh.V) > 3 and self.heap.size() > 0):
            # while ho is dirty compute cost and push back
            while(ho.e.dirty):
                v,c,Q = ho.e.computeCost()
                ho.e.vnew = ms.Vertex(v)
                ho.e.vnew.Q = Q
                ho.e.dirty = False
                self.heap.push(c,ho.e)
                ho = self.heap.pop()

            if(ho.c > .01): break
            if(fig is not None):
                fig.init('Simplification')
                #self.plotHeapCost(fig.ax1,ho, .0001, .015)
                self.plotHeapCovariances(fig.ax1,ho)
                mesh.draw(fig.ax1)
                fig.save('img_out/simple_')

            mesh.collapse(ho.e)
            ho = self.heap.pop()

    def plotHeapCovariances(self, ax, hoc):
        for ho in self.heap.h:
            C = ho.e.v1.cov()
            plotCov(ho.e.v1.p, 10.*C, ax)
        C = hoc.e.v1.cov()
        plotCov(hoc.e.v1.p, 10.*C, ax)

    def plotHeapCost(self, ax, hoc, cmin, cmax):
        lines = []
        costs = []
        lcmin = log10(cmin)
        lcmax = log10(cmax)
        print "plotting heap (size:",len(self.heap.h)+1,") costs..."
        for ho in self.heap.h:
            lines.append([(ho.e.v1.x(),ho.e.v1.y()), (ho.e.v2.x(),ho.e.v2.y())])
            if ho.e.dirty:
                v,c,Q = ho.e.computeCost()
            else: c = ho.c
            costs.append(log10(c))

        lines.append([(hoc.e.v1.x(),hoc.e.v1.y()), (hoc.e.v2.x(),hoc.e.v2.y())])
        costs.append(log10(hoc.c))
        costs = [ min(max(ci,lcmin),lcmax) for ci in costs]
        costs.append(lcmin)
        costs.append(lcmax)

        lc = LineCollection( lines, linewidths=(5.), cmap=plt.cm.jet)
        lc.set_array(array(costs))
        ax.add_collection(lc)
