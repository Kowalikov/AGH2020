import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys

def zrPlot(spx, spy, spz, uplim):
    fig, axs = plt.subplots(1, 1, figsize=(20, 12), sharex=True, sharey=True,
                            tight_layout=True)
    spr=np.sqrt(np.add(np.multiply(spx,spx),np.multiply(spy,spy)))
    axs.plot(spz, spr, "+")
    axs.set_xlim(-3000,3000)
    axs.set_ylim(0, 600)
    axs.set_title("Zbiorowy")
    #axs.legend()
    xmajor_ticks = np.arange(-3000, 3001, 1000)
    xminor_ticks = np.arange(-3000, 3001, 200)
    ymajor_ticks = np.arange(0, 601, 100)
    yminor_ticks = np.arange(0, 601, 20)

    axs.set_xticks(xmajor_ticks)
    axs.set_xticks(xminor_ticks, minor=True)
    axs.set_yticks(ymajor_ticks)
    axs.set_yticks(yminor_ticks, minor=True)
    axs.grid( which='both', axis='both', color='grey', linestyle='-')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    plt.savefig("./charts/dets_grid.png")
    plt.show()

spxTemp= []
spyTemp=[]
spzTemp=[]
spx=[]
spy=[]
spz=[]
rangeRead=1

for x in range(rangeRead):
    f = open("./data/raw_data/spx"+str(x)+".0")
    x0=0
    for line in f:
        spxTemp.append(line.split(" "))
        for x1 in range(len(spxTemp[x0][:-2])):
            spx.append(float(spxTemp[x0][x1]))
        x0+=1
for x in range(rangeRead):
    f = open("./data/raw_data/spy"+str(x)+".0")
    x0=0
    for line in f:
        spyTemp.append(line.split(" "))
        for x1 in range(len(spyTemp[x0][:-2])):
            spy.append(float(spyTemp[x0][x1]))
        x0+=1
for x in range(rangeRead):
    f = open("./data/raw_data/spz"+str(x)+".0")
    x0=0
    for line in f:
        spzTemp.append(line.split(" "))
        for x1 in range(len(spzTemp[x0][:-2])):
            spz.append(float(spzTemp[x0][x1]))
        x0+=1


zrPlot(spx, spy, spz, -1)


print("finito")
