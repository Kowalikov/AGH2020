import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numba import cuda
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PI = 3.14159

def simplePlot(spx, spy, uplim):
    fig, axs = plt.subplots(1, 1, figsize=(40, 30), sharex=True, sharey=True,
                            tight_layout=True)
    axs.plot(spx[:uplim], spy[:uplim], "+")
    axs.set_xlim(-52, -50)
    axs.set_ylim(-2, 0)

    xmajor_ticks = np.arange(-52, -50.2, 0.2)
    xminor_ticks = np.arange(-52, -50.05, 0.05)
    ymajor_ticks = np.arange(-2, 0.2, 0.1)
    yminor_ticks = np.arange(-2, 0.05, 0.05)

    axs.set_xticks(xmajor_ticks)
    axs.set_xticks(xminor_ticks, minor=True)
    axs.set_yticks(ymajor_ticks)
    axs.set_yticks(yminor_ticks, minor=True)

    axs.grid( which='both', axis='both', color='grey', linestyle='-')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    plt.savefig("./charts/xyPrecise.png")
    plt.show()

def houghTransformCircles(spx, spy, plot=False):
    size = len(spx)
    print(size)
    Aa = []
    Ab = []
    Ar = []
    rstep = 25
    rstart = 7*100
    rlim = 30*100
    rits=int((rlim-rstart)/rstep)
    for r in range(rstart, rlim, rstep):
        for x1 in range(size):
            for t in range(0, 360, 5):
                a = spx[x1] - r * np.cos(t * PI / 180)
                b = spy[x1] - r * np.sin(t * PI / 180)
                Aa.append(a)
                Ab.append(b)
                Ar.append(r)

    print(Aa, Ab, Ar, sep="\n")
    st = int(len(Aa) / rits)
    print("Zaczynamy pętłę", st)
    for x0 in range(rits):
        xt = Aa[x0 * st:((x0 + 1) * st - 1)]
        yt = Ab[x0 * st:((x0 + 1) * st - 1)]
        plt.plot(xt, yt, "+")
        tit = "R=" + str((rstart+x0*rstep)/100)
        plt.title(tit)
        plt.savefig("./charts/Transform/"+tit+".png")
        plt.show()
        """
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True,
                                tight_layout=True)
        axs.hist2d(xt, yt, bins=(30, 30), norm=colors.LogNorm())
        axs.grid()
        tit = "R=" + str((rstart+x0*rstep)/100)
        axs.set_title(tit)
        plt.show()
        """

def inverseHoughTransformCircles(spx, spy, plot=False):
    size = len(spx)
    print(size)
    Ad = []
    At = []
    Ar = []
    Ax = []
    tit=2
    rstep = 1
    rstart = 15*100
    rlim = 23*100
    rits=int((rlim-rstart)/rstep)
    rrange=[]
    r0 = np.zeros((int((rlim - rstart) / rstep), int(3600/tit)))
    x0=0
    for r in range(rstart, rlim, rstep):
        for t in range(0,3600,tit):
            for x1 in range(size):
                if 90 < abs(spx[x1]) or abs(spy[x1]) > 60:
                    a = r * np.cos(t * PI / 1800)
                    b = r * np.sin(t * PI / 1800)
                    diff=np.sqrt((a-spx[x1])**2+(b-spy[x1])**2)-r
                    if abs(diff)<10:
                        Ad.append(abs(diff))
                        At.append(t)
                        Ar.append(r)
                        Ax.append(x1)
                        if abs(diff)<0.1:
                            r0[x0][int(t/tit)]+=10
                        else:
                            r0[x0][int(t / tit)] += 1/(abs(diff))
        x0+=1


    print(Ad, At, Ar, Ax, sep="\n")
    print(r0)

    plt.figure(figsize=(6,8))
    ax = plt.gca()
    im = ax.imshow(r0, cmap='hot', interpolation='nearest', extent = [0 , 360, int(rstart) , int(rlim)])
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

    r0[r0 <= 12] = 0
    plt.figure(figsize=(6,8))
    ax = plt.gca()
    im = ax.imshow(r0, cmap='hot', interpolation='nearest', extent = [0 , 360, int(rstart) , int(rlim)])
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    """
    st = int(len(Aa) / rits)
    print("Zaczynamy pętłę", st)
    for x0 in range(rits):
        xt = Aa[x0 * st:((x0 + 1) * st - 1)]
        yt = Ab[x0 * st:((x0 + 1) * st - 1)]
        plt.plot(xt, yt, "+")
        tit = "R=" + str((rstart+x0*rstep)/100)
        plt.title(tit)
        plt.savefig("./charts/Transform/"+tit+".png")
        plt.show()
        '''
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True,
                                tight_layout=True)
        axs.hist2d(xt, yt, bins=(30, 30), norm=colors.LogNorm())
        axs.grid()
        tit = "R=" + str((rstart+x0*rstep)/100)
        axs.set_title(tit)
        plt.show()
        '''
    """

@cuda.jit
def inverseHoughTranformLinesKernel(spx, spy, spz, spr, RXlims, r0, rx):
    tx = cuda.threadIdx.x
    gx = cuda.blockIdx.x
    bx = cuda.blockDim.x
    #thr_id = tx + gx * bx

    pi = 3.14159

    size_dev = len(spx)
    tit=RXlims[0] #2
    tstart = RXlims[1]
    ystep =RXlims[3] #5
    ystart =RXlims[4]
    accuracy = RXlims[7]

    y0=ystart+tx*ystep
    t=tstart+gx*tit
    x0=0
    for x1 in range(size_dev):
        if 250<abs(spz[x1]) or spr[x1]>100:
            red_dist=(spz[x1]**2+spr[x1]**2)**0.5/500
            if red_dist>2:
                red_dist=2+red_dist*0.1
            y_est = math.tan(t*2*pi/(360*accuracy))*spz[x1]+y0/accuracy
            x_est =  (spr[x1] - y0/accuracy )/(math.tan(t * 2 * pi / (360*accuracy))+0.00001)
            diff=((y_est-spr[x1])**2+(x_est-spz[x1])**2)**0.5/red_dist
            if diff<6:
                rx[tx][gx][x0]=x1
                x0+=1
                if diff<0.5:
                    r0[tx][gx]+=2
                else:
                    r0[tx][gx] += 1/(abs(diff))

def inverseHoughTransformLines(spx, spy, spz, spr, entry, RXlims=[2,0,1800,5,-500,500, 4.0, 10], plot=False, plotMax=True, toFile=True):
    size = len(spx)
    #print(size)


    tit=RXlims[0] #2
    tstart = RXlims[1]
    tlim = RXlims[2]
    ystep =RXlims[3] #5
    ystart =RXlims[4]
    ylim = RXlims[5]
    accuracy = RXlims[7]
    yits=int((ylim-ystart)/ystep)
    tits=int(tlim-tstart/tit)

    # Create the data array - usually initialized some other way
    block = (yits)
    grid = (tits)

    r0 = np.zeros((yits, tits))
    rx = np.ones((yits, tits, 15))*(-1)
    RXlims = np.array(RXlims)
    inverseHoughTranformLinesKernel[grid, block](spx, spy, spz, spr, RXlims, r0, rx)

    plt.imshow(r0)
    plt.title("Entry: "+str(entry))
    plt.show()
    #print(r0)
    #print(rx)

    r1 = np.flipud(r0)
    rx = np.flipud(rx)

    if plot:
        plt.figure(figsize=(12,40))
        ax = plt.gca()
        im = ax.imshow(r1, cmap='hot', interpolation='nearest', extent = [0 , 180, int(ystart) , int(ylim)])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.title("Modified Hough Transform for event: "+str(entry), loc="right")
        plt.show()

    treshhold=RXlims[6] #5.0
    r1[r1 <= treshhold] = 0
    r1[r1 > treshhold] -= treshhold

    if plotMax:
        plt.figure(figsize=(12,40))
        ax = plt.gca()
        im = ax.imshow(r1, cmap='hot', interpolation='nearest', extent = [0 , 180, int(ystart) , int(ylim)])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.title("Modified Hough Transform for event: "+str(entry)+" with treshhold:"+str(treshhold), loc="right")
        plt.show()

    mR, mT = np.where(r1 > 0)
    mV = r1[np.where(r1 > 0)]
    mR = (ylim/accuracy - 1 - mR*ystep/accuracy)/1
    mT= (mT*tit)/accuracy
    #print("R:\t\t", mR, "\ntheta:\t",mT,"\nValues:\t",mV)

    if toFile==True:
        print("zapisuję do pliku")
        f = open("./data/InvertedLine/params"+str(entry)+".txt", mode="w")
        for i in range(len(mR)):
            f.write(str(mR[i]))
            f.write(";")
        f.write("\n")
        for i in range(len(mT)):
            f.write(str(mT[i]))
            f.write(";")
        f.write("\n")
        for i in range(len(mV)):
            f.write(str(mV[i]))
            f.write(";")
        f.write("\n")
        for i in range(len(RXlims)):
            if i==6:
                f.write(str(RXlims[i]))
            else:
                f.write(str(int(RXlims[i])))
            f.write(";")
        f.write("\n")
        f.write(str(len(r1))+";")
        f.write("\n")
        for i0 in range(len(r1)):
            for i1 in range(len(r1[i0])):
                f.write(str(r1[i0][i1]))
                f.write(";")
            f.write("\n")
        for i0 in range(len(rx)):
            for i1 in range(len(rx[0])):
                for i2 in range(len(rx[i0][i1])):
                    if rx[i0][i1][i2]>-0.5:
                        f.write(str(int(rx[i0][i1][i2])))
                        f.write(":")
                f.write(";")
            f.write("\n")
        f.close()
        print("\r\rZapisano do pliku")

    return mR, mT, mV, r1, rx, RXlims #chwilowo zwracamy różne maxima do podglądu

def houghTransformLine(spx,spy, spz, entry):
    size = len(spx)
    print(size)
    spr=np.sqrt(np.multiply(spx,spx)+np.multiply(spy,spy))

    At = []
    Ar = []

    tstep = 0.01
    tstart = 0
    tlim = 180
    tits=int((tlim-tstart)/tstep)

    po=0
    qpo=0
    for x1 in range(size):
        for t in np.linspace(tstart, tlim,num=int(tlim/tstep)-1):
            if (50<spz[x1] or spz[x1]<-50) and spr[x1]>50:
                r = spz[x1]*np.cos(t) + spr[x1]*np.sin(t)
                po+=1
                if -50<r<50:
                    At.append(t)
                    Ar.append(r)
                    qpo+=1
    """
    print(At, Ar, sep="\n")
    plt.plot(Ar, At, "+")
    plt.title("Straight transform for event: "+str(entry))
    #plt.savefig("./charts/Transform/"+tit+".png")
    plt.show()"""

    print("Calculation finished, time for hist")
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True,
                            tight_layout=True)

    #cmap=plt.cm.BuPu
    h2, xe, ye, img1 = axs.hist2d(Ar, At, bins=(100, 135), norm=matplotlib.colors.LogNorm())
    axs.grid()
    #axs.set_xlim(-300, 300)
    axs.set_title("Hist for event: "+str(entry))
    plt.show()


    plt.figure(figsize=(6, 8))
    ax = plt.gca()
    im = ax.imshow(h2, cmap='hot', interpolation='nearest', extent=[-50, 50, 0, 180])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

    h2[h2 <= 54] = 0
    plt.figure(figsize=(6, 8))
    ax = plt.gca()
    im = ax.imshow(h2, cmap='hot', interpolation='nearest', extent=[-50, 50, 0, 180])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

    print(h2)
    po=po*tstep/(tlim-tstart)
    qpo=qpo*tstep/(tlim-tstart)
    print("Po:", po, "qpo:", qpo)

def plotEvent(spx, spy, spz, event, TD=True):
    #fig = make_subplots(    rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]],
    #    subplot_titles=("First Subplot", "Second Subplot", "Third Subplot"))

    ax1 = plt.subplot(221)
    ax1.plot(spx, spy, "+")

    xmajor_ticks = np.arange(-600, 601, 100)
    xminor_ticks = np.arange(-600, 601, 10)
    ymajor_ticks = np.arange(-600, 601, 100)
    yminor_ticks = np.arange(-600, 601, 10)
    ax1.set_xticks(xmajor_ticks)
    ax1.set_xticks(xminor_ticks, minor=True)
    ax1.set_yticks(ymajor_ticks)
    ax1.set_yticks(yminor_ticks, minor=True)
    ax1.grid(which='both', axis='both', color='grey', linestyle='-')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)

    ax1.set_title("Event:"+str(event)+" spx-spy view")
    #plt.show()


    #figsize=(21, 3) for full zoom
    #fig, axs = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True,
    #                        tight_layout=True)

    ax2=plt.subplot(212)
    ax2.plot(spz, np.sqrt(np.multiply(spx,spx)+np.multiply(spy,spy)), "+")

    xmajor_ticks = np.arange(-3000, 3001, 500)
    xminor_ticks = np.arange(-3000, 3001, 50)
    ymajor_ticks = np.arange(0, 550, 100)
    yminor_ticks = np.arange(0, 550, 20)
    ax2.set_xticks(xmajor_ticks)
    ax2.set_xticks(xminor_ticks, minor=True)
    ax2.set_yticks(ymajor_ticks)
    ax2.set_yticks(yminor_ticks, minor=True)
    ax2.grid( which='both', axis='both', color='grey', linestyle='-')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)

    ax2.set_title("spz-spr view")
    #plt.show()

    #fig = plt.figure()
    ax3 = plt.subplot(222, projection='3d')
    ax3.scatter(spx, spy, spz, c='skyblue', s=60)
    ax3.set_ylabel("spy")
    ax3.set_xlabel("spx")
    ax3.set_zlabel("spz")
    ax3.set_title("spx-spy-spz view")
    #ax3.view_init(30, 185)"""

    #plt.figure(figsize=(5,10))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    #plt.title("Event:"+str(event))
    plt.subplots_adjust(
    top = 0.966,
    bottom = 0.038,
    left = 0.033,
    right = 0.985,
    hspace = 0.132,
    wspace = 0.025)
    plt.show()

def loadEvent(entry):
    spxTemp= []
    spyTemp=[]
    spzTemp=[]
    spx=[]
    spy=[]
    spz=[]
    f = open("./data/raw_data/spx"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0==entry%200:
            spxTemp.append(line.split(" "))
            for x1 in range(len(spxTemp[0][:-2])):
                spx.append(float(spxTemp[0][x1]))
        x0+=1
    f = open("./data/raw_data/spy"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0 == entry % 200:
            spyTemp.append(line.split(" "))
            for x1 in range(len(spyTemp[0][:-2])):
                spy.append(float(spyTemp[0][x1]))
        x0+=1
    f = open("./data/raw_data/spz"+str((entry - entry%200)/200))
    x0=0
    for line in f:
        if x0 == entry % 200:
            spzTemp.append(line.split(" "))
            for x1 in range(len(spzTemp[0][:-2])):
                spz.append(float(spzTemp[0][x1]))
        x0+=1
    print("Data Loaded. SPx.len: ", len(spx))
    spr=np.sqrt(np.multiply(spx,spx)+np.multiply(spy,spy))
    spx=np.array(spx)
    spy=np.array(spy)
    spz=np.array(spz)
    return spx, spy, spz, spr

"""
entry = 353
spx, spy, spz, spr = loadEvent(entry)
#mR,mT,mV, r1, rx, RXlims = \
inverseHoughTransformLines(spx, spy, spz, spr, entry, RXlims=[2,0,1800,2,-500,500, 3.5, 10], toFile=False)
"""


