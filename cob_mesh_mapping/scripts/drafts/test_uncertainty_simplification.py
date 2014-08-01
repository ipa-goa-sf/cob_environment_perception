
#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
import sensor_model as sm
import mesh_structure as ms
import mesh_optimization as mo

class Figure:
    def __init__(self):
        self.fig = plt.figure()
        self.c = 0
        self.xx1 = (-1.5,3.5)
        self.xx2 = (1.,3.)
        self.X1,self.X2 = meshgrid(linspace(self.xx1[0],self.xx1[1],50),
                                   linspace(self.xx2[0],self.xx2[1],50))

    ''' set up basic figure '''
    def init(self, title):
        self.fig.clf()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.grid()
        plt.title(title)

    ''' save current figure to file '''
    def save(self, filename, short=""):
        self.fig.savefig("uncertainty/mesh_"+str(self.c).zfill(5)+short+'.png')
        self.c = self.c + 1

    def plotPrecision(self,Q):
        self.f = zeros(shape(self.X1))
        for yi in range(shape(self.X1)[0]):
            for xi in range(shape(self.X1)[1]):
                x = mat([[self.X1[yi,xi]],[self.X2[yi,xi]],[1.]])
                self.f[yi,xi] = exp(-0.5*x.T*Q*x) * 1000.
        self.ax1.imshow(self.f,extent=[self.xx1[0],self.xx1[1],self.xx2[0],self.xx2[1]],
                        origin='lower',cmap=plt.cm.YlOrRd)



def weight(x,s):
    p = mat([[0],[0],[x]])
    return 100.*s.covariance(p)[-1,-1]

def computeQ(x1,x2,s):
    return linalg.inv( x1*x1.T/weight(x1[1],s) +
                       x2*x2.T/weight(x2[1],s) + 0.1*identity(3) )
plt.close('all')
nn = 40
sensor = sm.SensorModel()
beta_true = mat([.7,.7,-1.5]).T
beta_true = beta_true/linalg.norm(beta_true)
samples = mat(vstack([linspace(-1,3,nn),zeros([nn]),ones([nn])])).T
for i in range(int(.5*nn)):
    samples[i,1] = samples[i]*beta_true / -beta_true[1]
    samples[i,1] = samples[i,1] + weight(samples[i,1],sensor)*random.randn()

beta_true = mat([-1.,1.2,-0.25]).T
beta_true = beta_true/linalg.norm(beta_true)
for i in range(int(.5*nn),nn):
    samples[i,1] = samples[i]*beta_true / -beta_true[1]
    samples[i,1] = samples[i,1] + weight(samples[i,1],sensor)*random.randn()

m = ms.Mesh()
v1 = m.add(samples[0,0],samples[0,1])
v1.Q = mat(1.*computeQ(samples[0].T, samples[1].T, sensor) + 100.*identity(3))
v1.P = sensor.covariance(mat([[0],[v1.x],[v1.y]]))[1:,1:]
for i in range(1,nn):
    v2 = m.add(samples[i,0],samples[i,1])
    # P: samples covariance of sensor meassurement
    v2.P = sensor.covariance(mat([[0],[v2.x],[v2.y]]))[1:,1:]
    # Q: parameters precision of fit to samples [i-1] and [i]
    Q = computeQ(samples[i-1].T, samples[i].T, sensor)
    v1.Q = mat(v1.Q + Q)
    v2.Q = mat(v2.Q + Q)
    e = m.connect(v1,v2)
    e.dirty = True
    v1 = v2

v1.Q = mat(computeQ(samples[-1].T, samples[-2].T, sensor) + 100.*identity(3))

fig1 = Figure()
#plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.grid()

simpler = mo.Simplifier(m)

h = simpler.heap.pop()
fii = 0
while(len(m.V) > 3 and simpler.heap.size() > 0):
    while(h.data.dirty):
        # precision matrix (parameter space):
        Q = h.data.v1.Q + h.data.v2.Q
        # new covariance matrix (parameter space)
        Qinv = linalg.inv(Q)
        v = Qinv[:-1,-1]/Qinv[-1,-1]
        # covariance matrix (point space)
        P = Qinv[:-1,:-1] - v * Qinv[-1,:-1] + 1.*(v1.P+v2.P)
        h.data.vnew = ms.Vertex(float(v[0]),float(v[1]))
        h.data.vnew.Q = Q
        h.data.vnew.P = P
        c = float(v.T * P * v)
        print c
        h.data.dirty = False
        simpler.heap.push(c,h.data)
        h = simpler.heap.pop()

    if (h.cost < .0000001): break

    fig1.init('Mesh Simplification')
    m.draw(fig1.ax1, 've', 'bbb')
    fig1.plotPrecision(h.data.v1.Q + h.data.v2.Q)
    fig1.save('img_out/mesh_learner_')
    fii = fii+1

    m.collapse(h.data)
    h = simpler.heap.pop()

#simpler.simplifyAndPrint(1.,fig1)

#fig1.init('blub')
#fig1.ax1.plot(samples[:,0],samples[:,1],'x-')


#plt.show()
