#!/usr/bin/python
from numpy import *
import heapq
from collections import namedtuple
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import mesh_structure as ms


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
                self.plotHeapCost(fig.ax1)
                mesh.draw(fig.ax1)
                fig.save('img_out/simple_')

            mesh.collapse(ho.e)
            ho = self.heap.pop()

    def plotHeapCost(self, ax):
        lines = []
        costs = []
        for ho in self.heap.h:
            lines.append([(ho.e.v1.x(),ho.e.v1.y()), (ho.e.v2.x(),ho.e.v2.y())])
            if ho.e.dirty:
                v,c,Q = ho.e.computeCost()
            else: c = ho.c
            costs.append(log10(c))
        lc = LineCollection( lines, linewidths=(5.), cmap=plt.cm.jet )
        lc.set_array(array(costs))
        #ax.set_xlim(..)
        #ax.set_ylim(..)
        ax.add_collection(lc)
