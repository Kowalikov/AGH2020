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

def simplePlot(spx, spy, uplim, i):
    fig, axs = plt.subplots(1, 1, figsize=(20, 12), sharex=True, sharey=True,
                            tight_layout=True)
    axs.plot(spx[i][:uplim], spy[i][:uplim], "+")
    # plt.xlim(-150,150)
    # plt.ylim(0, 80)
    axs.set_xlim(-3000,3000)
    axs.set_ylim(-600, 600)
    axs.set_title("Zbiorowy")
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


for event in Tree:
    counter = 2200
    #if counter<=x<(counter+40):
    if x==2244:
        spzTemp=np.add(event.spZ, 0)
        spz.append(spzTemp)
        sprTemp=[]
        for i0 in range(len(event.spX)):
            sprTemp.append(np.sqrt(event.spY[i0]**2+event.spX[i0]**2))
        spr.append(sprTemp)
        simplePlot(spz, spr, -1, x)
    x += 1




step = 20000
steps = 8
print("finito")
#plotPath(spz, spr, steps, step, 0)
#simplePlot(spz, spr, -1, 0)
#for x1 in range(1):
#    plotHist(spz,spr, x1)



#treshhold cutoff


