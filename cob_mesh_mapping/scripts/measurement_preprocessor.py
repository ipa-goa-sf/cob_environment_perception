from numpy import *
import matplotlib.pyplot as plt
import mesh_structure as ms
import mesh_optimization as mo
import measurement_data as md

class Preprocessor:
    def __init__(self):
        self.mesh = ms.Mesh()
        self.simpler = mo.Simplifier()
        self.simpler.mesh = self.mesh

    '''converts sensor measurements to a compressed surface repressentation'''
    def compress(self, m, eps = 0.1):
        self.mesh = ms.Mesh()
        self.extendMesh(m)

        self.simpler = mo.Simplifier()
        self.simpler.mesh = self.mesh
        self.simplifyMesh(eps)

    def extendMesh(self, m):
        ii = len(m[:,0])
        for i in range(ii):
            if not math.isnan(m[i][0]): break

        v1 = self.mesh.add(m[i][0],m[i][1])
        broke = False
        for j in range(i+1,ii):
            if math.isnan(m[j][0]):
                broke = True
                continue

            v2 = self.mesh.add(m[j][0],m[j][1])
            # perform rough distance check:
            if not broke and fabs(m[j][0] - m[j-1][0]) < .04*(m[j][0])**2+.01:
                e = self.mesh.connect(v1,v2)
                e.dirty = True # mark as new

            broke = False
            v1 = v2

    def simplifyMesh(self, eps):
        for e in self.simpler.mesh.E:
            e.updateQuadricsAreaWeighted()
            #e.updateQuadrics()
        for e in self.simpler.mesh.E:
            self.simpler.markForUpdate(e)

        self.simpler.simplify2(eps)
