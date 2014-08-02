#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from mesh_structure import *

V1 = [Vertex([1.,1.]),
      Vertex([2.,1.]),
      Vertex([3.,2.]),
      Vertex([4.,.5])]

V2 = [Vertex([2.,1.]),
      Vertex([3.,2.]),
      Vertex([4.,.5]),
      Vertex([3., 0])]


lines = []
for i in range(len(V1)):
    lines.append( [(V1[i].x(), V1[i].y()), (V2[i].x(), V2[i].y())] )

lc = LineCollection( lines, linewidths=(5.), cmap=plt.cm.jet )
lc.set_array(array([5.,3.,10.,1.]))
ax = plt.axes()
ax.set_xlim((0.,5))
ax.set_ylim((-1.,3))
ax.add_collection(lc)
#axcb = plt.colorbar(lc)
plt.show()
