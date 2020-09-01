import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter

def animPlot(spx, spy, step = 8000, steps = 8):
    fig, ax = plt.subplots()
    ln1, = plt.plot([], [], '+')

    def init():
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)

    def update(i):
        ln1.set_data(spx[0][(i * step):(i + 2) * step], spy[0][(x * step):(x + 2) * step])

    ani = FuncAnimation(fig, update, range(9), init_func=init)
    plt.show()

def plotPath(spx, spy, R, ran, i):
    for x in range(R):
        plt.plot(spx[i][(x*ran):(x+2)*ran], spy[i][(x*ran):(x+2)*ran], "+")
        plt.xlim(-600,600)
        plt.ylim(-600, 600)
        plt.show()

def simplePlot(spx, spy, uplim, i):
    plt.plot(spx[i][:uplim], spy[i][:uplim], "+")
    # plt.xlim(-150,150)
    # plt.ylim(0, 80)
    plt.title("Zbiorowy")
    plt.show()

def plotHist(spx, spy, i):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True,
                            tight_layout=True)
    # We can increase the number of bins on each axis
    axs.hist2d(spx[i], spy[i], bins=(600, 600), norm=colors.LogNorm())
    fig.legend()
    axs.grid()
    tit = "Event:"+str(i)
    axs.set_title(tit)
    plt.show()

inFile = ROOT.TFile.Open("./data/measurements.root", "READ")
Tree = inFile.Get("SPTrkNtuple")
x = 0
spx = []
spy = []
for event in Tree:
    x+=1
    if x<10:
        spx.append(event.spX)
        spy.append(event.spY)

print(len(spx[0]))
step = 8000
steps = 8

#plotPath(spx, spy, steps, step, 0)
#simplePlot(spx, spy, (steps+1)*step, 0)
#for x1 in range(5):
#    plotHist(spx,spy, x1)

animPlot(spx, spy)



#treshhold cutoff


