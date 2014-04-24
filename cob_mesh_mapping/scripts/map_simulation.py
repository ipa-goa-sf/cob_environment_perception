#!/usr/bin/python

#%load_ext autoreload
#%autoreload 2

from numpy import *
from collections import namedtuple
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import CohenSutherlandClipping as csclip
import camera_model as cm
from tf_utils import *
import normal_estimation
import mesh_structure as ms
import mesh_optimization as mo
import measurement_data as md
import measurement_preprocessor as mp
import scanline_rasterization as sl
import iterative_mesh_learner as iml


### BEGIN CLASS -- Map ###
class World:
    def __init__(self, coords):
        self.coords = coords

    def draw(self, axis):
        axis.plot(self.coords[:,0], self.coords[:,1],
                  '-', lw=2, alpha=0.2, color='black', ms=10)

### END CLASS -- Map ###

### BEGIN CLASS -- Sensor ###
class Sensor(cm.Camera2d):
    def __init__(self, position, orientation):
        cm.Camera2d.__init__(self, 49.0/180.0 * pi, 4., .8)
        cm.Camera2d.setPose(self, position, orientation)

    def measure(self, world):
        w = make_affine(world.coords)
        self.world = transform(self.tf_to_cam,w)
        w = transform(self.tf_to_unit_cube.dot(self.tf_to_cam), w)
        #disp(w)

        #w = vstack(v / v[-1] for v in w)
        #for i in range(len(w)):
        #    if w[i][-1] != 0:
        #        w[i] = w[i]/fabs(w[i][-1])

        #disp(w)

        # clip lines and sort for intersection computation:
        c = csclip.Clipper()
        ww = []
        scan = sl.ScanlineRasterization()

        for i in range(len(w)-1):
            # the problem with homogeneous coordinates on different sides
            # (w>0, w<0)
            # http://www.ccciss.info/ciss380/ch7/37clip.htm
            # 3 cases (w1/w2):
            # (+/+): just clip
            # (-/-): mirror, than clip
            # (+/-),(-/+): clip, mirror and clip again
            pass0 = pass1 = False
            if w[i][-1] > 0:
                pass0, p00, p01 = c.clip(w[i],w[i+1]) #(+/+)
                if w[i+1][-1] < 0:
                    pass1, p10,p11 = c.clip(-1.*w[i],-1.*w[i+1]) #(+/-)
            else:
                pass0, p00, p01 = c.clip(-1.*w[i],-1.*w[i+1]) #(-/-)
                if w[i+1][-1] > 0:
                    pass1, p10, p11 = c.clip(w[i],w[i+1]) #(-/+)

            if pass0:
                if p00[-1] != 0: p0 = p00/p00[-1]
                else: p0 = p00
                if p01[-1] != 0: p1 = p01/p01[-1]
                else: p1 = p01

                ww[len(ww):] = [p0, p1]
                scan.addEdge(p0,p1)

            if pass1:
                if p10[-1] != 0: p0 = p10/p10[-1]
                else: p0 = p10
                if p11[-1] != 0: p1 = p11/p11[-1]
                else: p1 = p11

                ww[len(ww):] = [p0, p1]
                scan.addEdge(p0,p1)

        ww = vstack(ww)
        x,y = scan.contour([-1.,1.,-1.,1.], [2./self.res,2./self.res])
        x = [ float('nan') if xi >= 1. else xi for xi in x ]
        x += random.randn(len(x)) * 0.005

        #self.axis = plt.figure().add_subplot(111)
        #self.axis.set_xlim(-1., 1.)
        #self.axis.set_ylim(-1., 1.)
        #self.axis.plot(w[:,0],w[:,1],'r')
        #self.axis.plot(ww[:,0],ww[:,1])
        #self.axis.plot(x,y,'x')
        #self.axis.grid()

        back = self.tf_to_frustum
        vst = vstack(zip(x,y,ones(len(x))))

        self.measurement = vstack(v/v[-1] for v in transform(back,vst))

    def showMeasurement(self):
        self.axis = plt.figure().add_subplot(111)
        self.axis.plot(self.measurement[:,0],self.measurement[:,1],'x')
        self.axis.plot(self.world[:,0],self.world[:,1],'r')

    def drawMeasurement(self, axis):
        transformed = transform(self.tf_to_world,self.measurement)
        axis.plot(transformed[:,0],transformed[:,1], 'xr')

### END CLASS -- Sensor ###

class Figure:
    def __init__(self):
        self.fig = plt.figure(figsize=(1024.0/80, 768.0/80), dpi=80)
        self.c = 0
        self.w = None
        self.s = None

    def setWorld(self, world):
        self.w = world

    def setActiveSensor(self, sensor):
        self.s = sensor

    def init(self, title):
        self.fig.clf()
        self.ax1 = self.fig.add_subplot(111)

        if self.w is not None:
            self.w.draw(self.ax1)
        if self.s is not None:
            self.s.drawFrustum(self.ax1)
            self.s.drawPose(self.ax1)

        self.ax1.axis('equal')
        self.ax1.set_xlim(-.5, 6.5)
        self.ax1.set_ylim(-.5, 4.5)
        self.ax1.grid()
        plt.title(title)

    def save(self, filename, short=""):
        self.fig.savefig(filename+str(self.c).zfill(5)+short+'.png')
        self.c = self.c + 1



###----------------------------------------------------------------------------
#     provide simulation data (world, sensors, measurements)
###----------------------------------------------------------------------------

# create world model:
angles = array(range(18))/18.0*(-pi)-pi
circle = array([[cos(angles[i]),sin(angles[i])] for i in range(len(angles))])

world = World(vstack(
    [[-100.0,0],[0,0],[0,4.0],[3.0,4.0],[3.0,3.5],[0.5,3.5],
     [0.5,0],[3.5,0],[3.5,1.5],[2.5,1.5],[2.5,1.6],[3.0,1.6],#]))#,
     circle*0.2 + [3.2,1.9],
     [3.4,1.6],[4.6,1.6],[4.6,1.5],[3.6,1.5],[3.6,0],[100.0,0]]))

# create sensors and measure world:
s1 = array([Sensor([4.0, 5.0],[-1.,-.5]),
            Sensor([5.0, 4.5],[-1.,-.5]),
            Sensor([6.0, 4.0],[-1.,-.5]),
            Sensor([6.0, 3.0],[-1.,-.5]),
            Sensor([5.0, 3.5],[-1.,-.5]),
            Sensor([4.0, 3.0],[-1.,-.5]),
            Sensor([3.0, 2.5],[-1.,-.5]),
            Sensor([2.0, 2.0],[-1.,-.5])])

circle_size = 12.0
angles = array(range(int(circle_size)))/circle_size*(3.0/2.0*pi)-3.*pi/4.
circle = array([[cos(angles[i]),sin(angles[i])] for i in range(len(angles))])
s2 = array([Sensor([1.5,1.3],[cos(angles[i]),sin(angles[i])])
            for i in range(len(angles))])

#sensors = s1[-2:]
sensors = hstack([s1,s2])
#for s in sensors: s.measure(world)

###----------------------------------------------------------------------------
#     process measurements and add to map
###----------------------------------------------------------------------------

learner = iml.IterativeMeshLearner()
preproc = mp.Preprocessor()
data = []
colors = 'ym'
iii = 0
#sensors = [sensors[1], sensors[10], sensors[12]]

#for s in sensors:
    # normal estimation:
    #ii = len(s.measurement[:,0]) # number of measurement points
    #ne = Normals2d(9, ii)
    #n = empty([ii,2])
    #for i in range(ii):
    #    n[i] = ne.computeNormal(s.measurement[:,0],s.measurement[:,1],i)

    # create mesh:
    #m = Mesh()
    #m.load(s.measurement[:,0],s.measurement[:,1],n[:,0],n[:,1])
    #s.axis = plt.figure().add_subplot(111)
    #m.draw(s.axis,'ven')

    # simplify mesh:
    #ms = Simplifier()
    #ms.init(m)
    #ms.simplify(1.0)

    #m.draw(s.axis,'ve', 'kr'+colors[iii%len(colors)])

    # convert simplified mesh to input format for map optimization:
    #data_new = mdata.convertMeshToMeasurementData(m,s)
    #learner.initMesh(data_new[0])
    #learner.addMeasurement(data_new[1])
    #learner.addMeasurement(data_new[2])
    #data[len(data):] = [data_new[0],data_new[1],data_new[2]]
    #iii = iii+1



###----------------------------------------------------------------------------
#     visualize results
###----------------------------------------------------------------------------

fig1 = Figure()
fig1.setWorld(world)
#fig2 = plt.figure(figsize=(1024.0/80, 768.0/80), dpi=80)
#ax2 = fig2.add_subplot(111)

#cm.drawPoses(sensors,ax1)
#cm.drawPoses(sensors,ax2)
#for s in sensors: s.showMeasurementInMap(ax1)
#for s in sensors: s.drawFrustum(ax1)

fi = 0
for s in sensors:
    print "Sensor " + str(fi+1)
    fig1.setActiveSensor(s)

    # 1st: measure world
    s.measure(world)

    fig1.init('Measurement')
    learner.mesh.draw(fig1.ax1, 've', 'bbb')
    s.drawMeasurement(fig1.ax1)
    fig1.save('img_out/mesh_learner_')
    print "saved measurement image..."

    # 2nd: Preprocess new measurement (meshing + qslim)
    preproc.compress(s.measurement, 0.0001)
    learner.addMeasurements(md.convertMeshToMeasurementData(preproc.mesh, s))

    fig1.init('Measurement Compressed')
    learner.mesh.draw(fig1.ax1, 've', 'bbb')
    for d in learner.data: d.draw(fig1.ax1)
    fig1.save('img_out/mesh_learner_')
    print "saved compression image..."

    # 3rd: Refine and compensate exsisting map
    learner.extendMesh(s.measurement, s)

    fig1.init('Mesh Refinement')
    learner.mesh.draw(fig1.ax1, 've', 'bbb')
    fig1.save('img_out/mesh_learner_')
    print "saved refinement image..."

    # 4th: simplify refined parts of map using all measurements (wqslim)

    learner.compensate(s)
    learner.prepareSimplification()
    learner.simpler.simplifyAndPrint(0.0001,fig1)
    print "saved simplification image..."
    fi = fi+1

#for d in data:
#    d.draw(ax2)
#    d.drawBoundingBox(ax2,[.1,.1])



#ax2.axis('equal')
#ax2.set_xlim(-.5, 6.5)
#ax2.set_ylim(-.5, 4.5)
#ax2.grid()

#plt.show()
