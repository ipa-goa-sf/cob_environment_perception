import matplotlib.pyplot as plt

def plotPointOnTriangle():
    tri = np.hstack(vpoints(py("*sf_map"),["vh1","vh2","vh3"], "proj_point"))
    p = vpoints(py("*sf_sensor"),["vh_sensor"],"proj_point")[0]
    plt.plot(tri[0],tri[1])
    plt.plot(p[0],p[1],'o')
    plt.show()
