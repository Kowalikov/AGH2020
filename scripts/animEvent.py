import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

PI = 3.14159

def plotEvent(spx, spy, spz, event):
    fig, axs = plt.subplots(1, 1, figsize=(12, 6.5), sharex=True, sharey=True,
                            tight_layout=True)
    axs.plot(spx, spy, "+")

    xmajor_ticks = np.arange(-600, 601, 100)
    xminor_ticks = np.arange(-600, 601, 10)
    ymajor_ticks = np.arange(-400, 250, 100)
    yminor_ticks = np.arange(-400, 250, 10)

    axs.set_xticks(xmajor_ticks)
    axs.set_xticks(xminor_ticks, minor=True)
    axs.set_yticks(ymajor_ticks)
    axs.set_yticks(yminor_ticks, minor=True)

    axs.grid(which='both', axis='both', color='grey', linestyle='-')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)

    plt.title("Event:"+str(event))
    plt.show()


    #figsize=(21, 3) for full zoom
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True,
                            tight_layout=True)
    axs.plot(spz, np.sqrt(np.multiply(spx,spx)+np.multiply(spy,spy)), "+")

    xmajor_ticks = np.arange(-500, 3001, 500)
    xminor_ticks = np.arange(-500, 3001, 50)
    ymajor_ticks = np.arange(0, 550, 100)
    yminor_ticks = np.arange(0, 550, 20)

    axs.set_xticks(xmajor_ticks)
    axs.set_xticks(xminor_ticks, minor=True)
    axs.set_yticks(ymajor_ticks)
    axs.set_yticks(yminor_ticks, minor=True)

    axs.grid( which='both', axis='both', color='grey', linestyle='-')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)

    plt.title("Event:"+str(event))
    plt.show()

def loadEvent(entry):
    spxTemp= []
    spyTemp=[]
    spzTemp=[]
    spx=[]
    spy=[]
    spz=[]
    f = open("../data/raw_data/spx"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0==entry%200:
            spxTemp.append(line.split(" "))
            for x1 in range(len(spxTemp[0][:-2])):
                spx.append(float(spxTemp[0][x1]))
        x0+=1
    f = open("../data/raw_data/spy"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0 == entry % 200:
            spyTemp.append(line.split(" "))
            for x1 in range(len(spyTemp[0][:-2])):
                spy.append(float(spyTemp[0][x1]))
        x0+=1
    f = open("../data/raw_data/spz"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0 == entry % 200:
            spzTemp.append(line.split(" "))
            for x1 in range(len(spzTemp[0][:-2])):
                spz.append(float(spzTemp[0][x1]))
        x0+=1
    print("Data Loaded. SPx.len: ", len(spx))
    return spx, spy, spz

def animPlot(spx, spy, steps = 20,):
    fig, ax = plt.subplots()
    ln1, = plt.plot([], [], '+')

    def init():
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)

    def update(i):
        step=int(np.floor(len(spx)/steps))
        ln1.set_data(spx[(i * step):(i + 4) * step], spy[(i * step):(i + 4) * step])
        ax.set_title(str(i))


    ani = FuncAnimation(fig, update, range(steps-4), init_func=init)
    #ani.save("anim.gif", fps=5)
    ani.show()

entry = 10
spx, spy, spz = loadEvent(entry)
#plotEvent(spx, spy, spz, entry)
#animPlot(spx, spy, steps=200)
steps=40
for i in range(steps-4):
    fig, ax = plt.subplots()
    ln1, = plt.plot([], [], '+')
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    step = int(np.floor(len(spx) / steps))
    ln1.set_data(spx[(i * step):(i + 4) * step], spy[(i * step):(i + 4) * step])
    ax.set_title(str(i))
    plt.show()

print(spx)


print("finito")

