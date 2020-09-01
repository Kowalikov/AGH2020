import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter

def plotPath(spx, spy, R, ran, i):
    for x in range(R):
        plt.plot(spx[i][(x*ran):(x+2)*ran], spy[i][(x*ran):(x+2)*ran], "+")
        plt.xlim(-600,600)
        plt.ylim(-600, 600)
        plt.show()

def simplePlot(spx, spy, uplim):
    fig, axs = plt.subplots(1, 1, figsize=(20, 12), sharex=True, sharey=True,
                            tight_layout=True)
    axs.plot(spx, spy, "+")
    # plt.xlim(-150,150)
    # plt.ylim(0, 80)
    axs.set_xlim(-3000,3000)
    axs.set_ylim(-600, 600)
    axs.set_title("Zbiorowy")
    axs.legend()
    plt.show()

def plotHist(spx, spy, i):
    fig, axs = plt.subplots(1, 1, figsize=(20, 12), sharex=True, sharey=True,
                            tight_layout=True)
    # We can increase the number of bins on each axis
    axs.hist2d(spx[i], spy[i], bins=(3000, 600), norm=colors.LogNorm())
    fig.legend()
    axs.grid()
    tit = "Event:"+str(i)
    axs.set_title(tit)
    plt.show()

def animation(spx, spy):
    fig, ax = plt.subplots()
    ln1, = plt.plot([], [], '+')

    def init():
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)

    def update(i):
        ln1.set_data(spx[0][(i * step):(i + 2) * step], spy[0][(x * step):(x + 2) * step])

    ani = FuncAnimation(fig, update, range(9), init_func=init)
    plt.show()

inFile = ROOT.TFile.Open("./data/measurements.root", "READ")
Tree = inFile.Get("SPTrkNtuple")
x = 0
spz = []
spr = []


sprTemp=[]
spzTemp=[]
for event in Tree:
    if x<10:
        for i0 in range(len(event.spX)):
            sprTemp.append(np.sqrt(event.spY[i0]**2+event.spX[i0]**2))
            spzTemp.append(np.add(event.spZ[i0], 0))
    x += 1
    print(x, len(sprTemp), len(spzTemp))

spr.append(sprTemp)
spz.append(spzTemp)

simplePlot(spz, spr, -1)



step = 20000
steps = 8
print("finito")
#plotPath(spz, spr, steps, step, 0)
#simplePlot(spz, spr, -1, 0)
#for x1 in range(1):
#    plotHist(spz,spr, x1)



#treshhold cutoff


