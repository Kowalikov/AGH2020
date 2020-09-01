import HoughTransform as HT
import HoughTransformGPU as HTG
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from upgradedLibraries.skimage_feature import peak_local_max
from math import isclose

def animTransformFit(spz, spr, mR, mT, mV):
    fig, ax = plt.subplots()
    ln1, = plt.plot([], [], '-')
    ln2, = plt.plot(spz, spr, '+')

    def init():
        ax.set_xlim(-500, 3000)
        ax.set_ylim(-200, 600) #add grid

    def update(i):
        ln1.set_data(np.linspace(-200, 3000, 201), np.linspace(-200, 3000, 201) * np.tan(mT[i] * 2 * np.pi / 360) + mR[i])
        ax.set_title("Line from y0=" + str(mR[i]) + " and t=" + str(mT[i]) + " maximum ("+str(np.ceil(100*i/len(mR))/100)+")")

    ani = FuncAnimation(fig, update, range(len(mR)), init_func=init)
    ani.save("animFit.gif", fps=5)#animation of all the non treshholded parameters

def loadInvTransParams(entry):
    f = open("./data/InvertedLine/params"+str(entry)+".txt", mode="r")
    mR=[]
    mT=[]
    mV=[]
    r1=[]
    rx=[]
    RXlims=[]
    x0=0
    r1size=0
    for line in f:
        temp = line.split(";")
        for x1 in range(len(temp)-1):
            if x0==0:
                mR.append(float(temp[x1]))
            elif x0==1:
                mT.append(float(temp[x1]))
            elif x0==2:
                mV.append(float(temp[x1]))
            elif x0 == 3:
                if x1==6:
                    RXlims.append(float(temp[x1]))
                else:
                    RXlims.append(int(temp[x1]))
            elif x0==4:
                r1size=int(temp[x1])
                for x2 in range(r1size):
                    r1.append([])
            elif x0<r1size+5:
                r1[x0-5].append(float(temp[x1]))
            else:
                temp1 = temp[x1].split(":")
                rx.append([])
                temp2=[]
                for x2 in range(len(temp1)-1):
                    if temp1[x2]!=[]:
                        temp2.append(int(temp1[x2]))
                    else:
                        temp2.append([])
                rx[x0-5-r1size].append(list(temp2))
        x0+=1
    f.close()
    r1 = np.array(r1)
    return mR,mT,mV, r1, rx, RXlims

def find_local_maxima(spx, spy, spz, r1, rx, RXlims, plot=False, treshhold=0.2):
    neighborhood_size = [int(np.ceil(5 * RXlims[7] / (RXlims[0]))),
                         int(np.ceil(2.5 * RXlims[7] / (RXlims[3])))]  # (theta, R) zasięg minimów to 15[deg] 2.5[m]
    rtheta_est_raw = peak_local_max(r1, size=neighborhood_size, num_peaks_per_label=1, threshold_rel=treshhold)
    rtheta_est_raw = np.transpose(rtheta_est_raw)
    #print("Rtheta est raw:", rtheta_est_raw)

    r_est = (RXlims[5] - rtheta_est_raw[0]*RXlims[3]) / RXlims[7]
    theta_est = (rtheta_est_raw[1]*RXlims[0]+RXlims[1]) / RXlims[7]
    #print("R:", r_est, "\ntheta", theta_est)

    spxClasd = []
    spyClasd = []
    spzClasd = []
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for i in range(len(rtheta_est_raw[0])):
        temp = np.array(rx[rtheta_est_raw[0][i]][rtheta_est_raw[1][i]])
        spxClasd.append([])
        spyClasd.append([])
        spzClasd.append([])
        for counter, value in enumerate(temp):
            if value >-0.5:
                spxClasd[i].append(spx[int(value)])
                spyClasd[i].append(spy[int(value)])
                spzClasd[i].append(spz[int(value)])
        if (plot):
            ax.scatter(spxClasd[i], spyClasd[i], spzClasd[i], label=str(i), s=40, alpha=0.8)

    if plot:
        ax.scatter(spx, spy, spz, c='skyblue', s=30, alpha=0.3)
        ax.set_ylabel("spy")
        ax.set_xlabel("spx")
        ax.set_zlabel("spz")
        ax.set_title("Fitted, labeled points")
        ax.view_init(30, 185)
        plt.legend()
        plt.show()

    return spxClasd, spyClasd, spzClasd, rtheta_est_raw, r_est, theta_est

def plot3D(spx, spy, spz, spxClasd, spyClasd, spzClasd, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(spxClasd)):
        ax.scatter(spxClasd[i], spyClasd[i], spzClasd[i], label=str(i), s=40, alpha=0.9)
    ax.scatter(spx, spy, spz, c='skyblue', s=30, alpha=0.5)
    ax.set_ylabel("spy")
    ax.set_xlabel("spx")
    ax.set_zlabel("spz")
    ax.set_title("Fitted, labeled points")
    ax.view_init(30, 185)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_found_maximum(spr, spz, r, theta):
    plt.plot(spz, spr, "+")
    for i0 in range(len(r)):
        plt.plot(np.linspace(-200, 3000, 201), np.linspace(-200, 3000, 201) * np.tan(theta[i0] * 2 * np.pi / 360) + r[i0], label=str(i0))
    plt.xlim(-3000, 3000)
    plt.ylim(0, 600)
    plt.grid()
    plt.title("Line from y0=" + str(r/10) + " and t=" + str(theta) + " maximum")
    plt.show()

def ReduceMaximaLessThan_Points(rtheta_est_raw, spxClasd, spyClasd, spzClasd, rx, points=4):
    rtheta_reduced = rtheta_est_raw
    spClasedReduced = np.array((spxClasd, spyClasd, spzClasd))

    for i in range(len(rtheta_est_raw[0]) - 1, -1, -1):
        if rx[rtheta_est_raw[0][i]][rtheta_est_raw[1][i]][points-1] == -1.0:
            rtheta_reduced = np.delete(rtheta_reduced, i, 1)
            spClasedReduced = np.delete(spClasedReduced, i, 1)

    return rtheta_reduced, spClasedReduced

def ReduceMaximaXYscatteredByDistDiff(  rtheta_reduced_same, spClasedReduced_same):
    spr=[]
    spphi=[]
    for x0 in range(len(spClasedReduced_same[0])):
        temp=[]
        tempphi=[]
        for x1 in range(len(spClasedReduced_same[0][x0])):
            temp.append(np.sqrt(spClasedReduced_same[0][x0][x1]**2+spClasedReduced_same[1][x0][x1]**2))
            if spClasedReduced_same[0][x0][x1]==0:
                if spClasedReduced_same[1][x0][x1]>0:
                    tempphi.append(np.pi/2)
                else:
                    tempphi.append(3*np.pi/2)
            elif spClasedReduced_same[0][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1])+np.pi)
            elif spClasedReduced_same[1][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]) + 2*np.pi)
            else:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]))
        spr.append(temp)
        spphi.append(tempphi)

    spx=spClasedReduced_same[0]
    spy=spClasedReduced_same[1]
    spz=spClasedReduced_same[2]

    #print(spClasedReduced_same, "\n", spr)
    for x0 in range(len(spx)):
        order = np.argsort(spr[x0][:], axis=0, kind='mergesort')
        spx[x0]=np.array(spx[x0])
        spx[x0] = spx[x0][order]
        spy[x0]=np.array(spy[x0])
        spy[x0] = spy[x0][order]
        spz[x0]=np.array(spz[x0])
        spz[x0] = spz[x0][order]
        spr[x0]=np.array(spr[x0])
        spr[x0] = spr[x0][order]
        spphi[x0]=np.array(spphi[x0])
        spphi[x0] = spphi[x0][order]

    #print(spx, len(spy), len(spr), sep="\n")
    for x0 in range(len(spx)-1,-1,-1):
        for x1 in range(len(spx[x0])-1,-1,-1):
            distR_prev=abs(spr[x0][x1]-spr[x0][x1-1])
            distR_next=abs(spr[x0][x1]-spr[x0][(x1+1)%(len(spr[x0])-1)])
            distXY_prev=abs(np.sqrt((spx[x0][x1]-spx[x0][x1-1])**2+(spy[x0][x1]-spy[x0][x1-1])**2))
            distXY_next=abs(np.sqrt((spx[x0][x1]-spx[x0][(x1+1)%(len(spx[x0])-1)])**2+(spy[x0][x1]-spy[x0][(x1+1)%(len(spy[x0])-1)])**2))
            #print("Maximum: ", x0, "\tpoint: ", x1, "/", len(spr[x0]), "\t2D: ", round(distR_prev, 2), "<- ->", round(distR_next,2), "      \t3D: ", round(distXY_prev,2), "<- ->", round(distXY_next,2), "     \t(X,Y,Z,R): (", round(spx[x0][x1],2), ";", round(spy[x0][x1],2), ";", round(spz[x0][x1],2), ";", round(spr[x0][x1],2), "\t)",sep="")
            if not (0.75<distR_prev/distXY_prev<1.25 and 0.75<distR_next/distXY_next<1.25):
                spx[x0] = np.delete(spx[x0], x1, 0)
                spy[x0] = np.delete(spy[x0], x1, 0)
                spz[x0] = np.delete(spz[x0], x1, 0)
                spr[x0] = np.delete(spr[x0], x1, 0)
                spphi[x0] = np.delete(spphi[x0], x1, 0)
            if len(spx[x0])<3:
                spx=np.delete(spx,x0,0)
                spy= np.delete(spy, x0, 0)
                spz= np.delete(spz, x0, 0)
                spr= np.delete(spr, x0, 0)
                spphi= np.delete(spphi, x0, 0)
                rtheta_reduced_same=np.delete(rtheta_reduced_same, x0, 1)
                break

    sp = np.array([spx,spy,spz,spr,spphi])
    #print(sp)

    return rtheta_reduced_same, sp
    """    dist2D_prev = abs(spr[x0][x1] - spr[x0][x1 - 1])
    dist2D_next = abs(spr[x0][x1] - spr[x0][(x1 + 1) % len(spr[x0])])
    dist3D_prev = abs(np.sqrt((spx[x0][x1] - spx[x0][x1 - 1]) ** 2 + (spy[x0][x1] - spy[x0][x1 - 1]) ** 2 + (
                spz[x0][x1] - spz[x0][x1 - 1]) ** 2))
    dist3D_next = abs(np.sqrt((spx[x0][x1] - spx[x0][(x1 + 1) % len(spr[x0])]) ** 2 + (
                spy[x0][x1] - spy[x0][(x1 + 1) % len(spr[x0])]) ** 2 + (
                                          spz[x0][x1] - spz[x0][(x1 + 1) % len(spr[x0])]) ** 2))
    print("Maximum: ", x0, "\tpoint: ", x1, "/", len(spr[x0]), "\t2D: ", round(dist2D_prev, 2), "<- ->",
          round(dist2D_next, 2), "      \t3D: ", round(dist3D_prev, 2), "<- ->", round(dist3D_next, 2),
          "     \t(X,Y,Z,R): (", round(spx[x0][x1], 2), ";", round(spy[x0][x1], 2), ";", round(spz[x0][x1], 2), ";",
          round(spr[x0][x1], 2), "\t)", sep="")
    """

def ReduceMaximaXYscatteredByAngleDiffUNFINISHED( rtheta_reduced_same, spClasedReduced_same):
    spr=[]
    spphi=[]
    for x0 in range(len(spClasedReduced_same[0])):
        temp=[]
        tempphi=[]
        for x1 in range(len(spClasedReduced_same[0][x0])):
            temp.append(np.sqrt(spClasedReduced_same[0][x0][x1]**2+spClasedReduced_same[1][x0][x1]**2))
            if spClasedReduced_same[0][x0][x1]==0:
                if spClasedReduced_same[1][x0][x1]>0:
                    tempphi.append(np.pi/2)
                else:
                    tempphi.append(3*np.pi/2)
            elif spClasedReduced_same[0][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1])+np.pi)
            elif spClasedReduced_same[1][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]) + 2*np.pi)
            else:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]))
        spr.append(temp)
        spphi.append(tempphi)

    spx=spClasedReduced_same[0]
    spy=spClasedReduced_same[1]
    spz=spClasedReduced_same[2]

    #print(spClasedReduced_same, "\n", spr)
    for x0 in range(len(spx)):
        order = np.argsort(spr[x0][:], axis=0, kind='mergesort')
        spx[x0]=np.array(spx[x0])
        spx[x0] = spx[x0][order]
        spy[x0]=np.array(spy[x0])
        spy[x0] = spy[x0][order]
        spz[x0]=np.array(spz[x0])
        spz[x0] = spz[x0][order]
        spr[x0]=np.array(spr[x0])
        spr[x0] = spr[x0][order]
        spphi[x0]=np.array(spphi[x0])
        spphi[x0] = spphi[x0][order]

    #print(spx, len(spy), len(spr), sep="\n")
    for x0 in range(len(spx)-1,-1,-1):
        temp=np.zeros(size=(2,8))
        for x1 in range(1,9):
            temp[0][x1-1]=np.average(np.mod(spphi[x0] + x1 * np.pi / 4, 2 * np.pi))
            temp[1][x1-1]=np.std(np.mod(spphi[x0] + x1 * np.pi / 4, 2 * np.pi))
        minT=np.argmin(temp[1])
        #jak nie wystarczy zminimalizowanie dewiacji dla cąłośći to można pomyślec o minimalizacji dla każdego punktu i dopiero póżniej odcinaniu największych
        minphase = np.argmin(np.abs(np.diff(np.mod(spphi[x0] + x1 * np.pi / 4,2*np.pi), temp[0][minT])))


        for x1 in range(len(spx[x0])-1,-1,-1):


            distR_prev = abs(spr[x0][x1] - spr[x0][x1 - 1])
            distR_next = abs(spr[x0][x1] - spr[x0][(x1 + 1) % (len(spr[x0]) - 1)])



            angle_prev=abs(spphi[x0][x1]-spphi[x0][x1-1])
            if angle_prev>np.pi:
                angle_prev=2*np.pi-angle_prev
                print("SPR: ", spr[x0][x1])
            angle_prev=angle_prev * 600/distR_prev

            angle_next=abs(spphi[x0][x1]-spphi[x0][(x1+1)%(len(spphi[x0])-1)])
            if angle_next>np.pi:
                angle_next=2*np.pi-angle_next
            angle_next = angle_next * 600 / distR_next

            #print("Maximum: ", x0, "\tpoint: ", x1, "/", len(spr[x0]), "\t2D: ", round(distR_prev, 2), "<- ->", round(distR_next,2), "      \t3D: ", round(distXY_prev,2), "<- ->", round(distXY_next,2), "     \t(X,Y,Z,R): (", round(spx[x0][x1],2), ";", round(spy[x0][x1],2), ";", round(spz[x0][x1],2), ";", round(spr[x0][x1],2), "\t)",sep="")
            if not (angle_prev<(np.pi/2) and angle_next<(np.pi/2)):
                spx[x0] = np.delete(spx[x0], x1, 0)
                spy[x0] = np.delete(spy[x0], x1, 0)
                spz[x0] = np.delete(spz[x0], x1, 0)
                spr[x0] = np.delete(spr[x0], x1, 0)
                spphi[x0] = np.delete(spphi[x0], x1, 0)
            if len(spx[x0])<3:
                spx=np.delete(spx,x0,0)
                spy= np.delete(spy, x0, 0)
                spz= np.delete(spz, x0, 0)
                spr= np.delete(spr, x0, 0)
                spphi= np.delete(spphi, x0, 0)
                rtheta_reduced_same=np.delete(rtheta_reduced_same, x0, 1)
                break

    sp = np.array([spx,spy,spz,spr,spphi])
    #print(sp)

    return rtheta_reduced_same, sp,

def ReduceMaximaXYscatteredByAngleDiffBySTD( rtheta_reduced_same, spClasedReduced_same, plot=False):
    spr=[]
    spphi=[]
    for x0 in range(len(spClasedReduced_same[0])):
        temp=[]
        tempphi=[]
        for x1 in range(len(spClasedReduced_same[0][x0])):
            temp.append(np.sqrt(spClasedReduced_same[0][x0][x1]**2+spClasedReduced_same[1][x0][x1]**2))
            if spClasedReduced_same[0][x0][x1]==0:
                if spClasedReduced_same[1][x0][x1]>0:
                    tempphi.append(np.pi/2)
                else:
                    tempphi.append(3*np.pi/2)
            elif spClasedReduced_same[0][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1])+np.pi)
            elif spClasedReduced_same[1][x0][x1]<0:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]) + 2*np.pi)
            else:
                tempphi.append(np.arctan(spClasedReduced_same[1][x0][x1] / spClasedReduced_same[0][x0][x1]))
        spr.append(temp)
        spphi.append(tempphi)

    spx=spClasedReduced_same[0]
    spy=spClasedReduced_same[1]
    spz=spClasedReduced_same[2]

    #print(spClasedReduced_same, "\n", spr)
    for x0 in range(len(spx)):
        order = np.argsort(spr[x0][:], axis=0, kind='mergesort')
        spx[x0]=np.array(spx[x0])
        spx[x0] = spx[x0][order]
        spy[x0]=np.array(spy[x0])
        spy[x0] = spy[x0][order]
        spz[x0]=np.array(spz[x0])
        spz[x0] = spz[x0][order]
        spr[x0]=np.array(spr[x0])
        spr[x0] = spr[x0][order]
        spphi[x0]=np.array(spphi[x0])
        spphi[x0] = spphi[x0][order]

    #print(spx, len(spy), len(spr), sep="\n")
    for x0 in range(len(spx)-1,-1,-1):
        temp=np.zeros(shape=(2,8))
        for x1 in range(0,8):
            temp[0][x1]=np.average(np.mod(spphi[x0] + x1 * np.pi / 4, 2 * np.pi))
            temp[1][x1]=np.std(np.mod(spphi[x0] + x1 * np.pi / 4, 2 * np.pi))
        minT=np.argmin(temp[1])
        avePhiShifted=temp[0][minT]
        spPhiShifted = np.mod(spphi[x0] + minT * np.pi / 4, 2 * np.pi)
        distPhi = np.abs(avePhiShifted - spPhiShifted)
        distXY = np.abs(np.sqrt(np.square(spx[x0]) + np.square(spy[x0])))

        ranID = np.where(distXY > 80)
        #print(spphi[x0], minT, avePhiShifted, spPhiShifted, distPhi, distXY, sep="\n", end="\n\n")
        #print(distXY, ranID, distXY[ranID[0]], spPhiShifted, spPhiShifted[ranID[0]])
        only_center = False
        if len(ranID[0])<2:
            only_center=True
            m0=0
            b0=0
        else:
            m0, b0 = np.polyfit(distXY[ranID[0]], spPhiShifted[ranID[0]], 1)
        #a, m, b, c = np.polyfit(distXY[ranID[0]], spPhiShifted[ranID[0]], 3)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.plot(spx[x0], spy[x0], '+')
            ax1.set_xlim(-600, 600)
            ax1.set_ylim(-600, 600)
            ax1.grid()
            plt.grid()

            ax2.set_xlim(left=0, right=900)
            ax2.set_ylim(0, 360)
            ax2.grid()
            ax2.plot(distXY, spPhiShifted * 180 / np.pi, '+')
            #ax2.plot(distXY, (a * (distXY ** 3) + m * (distXY ** 2) + b * distXY + c) * 180 / np.pi, '-')

            plt.grid()
            plt.show()

        deleted=False

        #print(m0, m0*600, m0*600*360/(2*np.pi))
        slope_treshhold=120*2*np.pi/(360*600) #maksymalnie koło z ostatnim punktem 120stopni(przeliczone na radiany) na długości detektora
        if abs(m0)>slope_treshhold or only_center:
            spx = np.delete(spx, x0, 0)
            spy = np.delete(spy, x0, 0)
            spz = np.delete(spz, x0, 0)
            spr = np.delete(spr, x0, 0)
            spphi = np.delete(spphi, x0, 0)
            rtheta_reduced_same = np.delete(rtheta_reduced_same, x0, 1)
            deleted = True
            #print("Deleted: ", x0)
        else:
            for x1 in range(len(spPhiShifted)-1,-1,-1):
                if distPhi[x1]>np.pi:
                    distPhi[x1]=2*np.pi-distPhi[x1]

                #print("Maximum: ", x0, "\tpoint: ", x1, "/", len(spr[x0]), "\t2D: ", round(distR_prev, 2), "<- ->", round(distR_next,2), "      \t3D: ", round(distXY_prev,2), "<- ->", round(distXY_next,2), "     \t(X,Y,Z,R): (", round(spx[x0][x1],2), ";", round(spy[x0][x1],2), ";", round(spz[x0][x1],2), ";", round(spr[x0][x1],2), "\t)",sep="")
                if (abs(spPhiShifted[x1]-m0*distXY[x1]-b0)>np.pi/6 and distXY[x1]>80):
                    spx[x0] = np.delete(spx[x0], x1, 0)
                    spy[x0] = np.delete(spy[x0], x1, 0)
                    spz[x0] = np.delete(spz[x0], x1, 0)
                    spr[x0] = np.delete(spr[x0], x1, 0)
                    spphi[x0] = np.delete(spphi[x0], x1, 0)
                if len(spx[x0])<3:
                    spx=np.delete(spx,x0,0)
                    spy= np.delete(spy, x0, 0)
                    spz= np.delete(spz, x0, 0)
                    spr= np.delete(spr, x0, 0)
                    spphi= np.delete(spphi, x0, 0)
                    rtheta_reduced_same=np.delete(rtheta_reduced_same, x0, 1)
                    deleted=True
                    #print("Deleted: ", x0)
                    break

        #print(spphi[x0], minT, avePhiShifted, spPhiShifted, distPhi, distXY, sep="\n", end="\n\n")
        #m, b= np.polyfit(distXY, spPhiShifted)

        if deleted==False and plot==True:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.plot(spx[x0], spy[x0], '+')
            ax1.set_xlim(-600, 600)
            ax1.set_ylim(-600, 600)
            ax1.grid()
            plt.grid()

            ax2.set_xlim(left=0, right=900)
            ax2.set_ylim(0, 360)
            ax2.grid()
            ax2.plot(np.abs(np.sqrt(np.square(spx[x0]) + np.square(spy[x0]))), spphi[x0] * 180 / np.pi, '+')

            plt.grid()
            plt.title("Cleared: "+str(x0))
            plt.show()

    sp = np.array([spx,spy,spz,spr,spphi])

    return rtheta_reduced_same, sp

def ReduceMaximaWithSamePoints(rtheta_reduced, spClasedReduced, r1):
    dels=[]
    for x0 in range(len(rtheta_reduced[0])-1):
        for x1 in range(x0+1,len(rtheta_reduced[0])):
            same=0
            for x2 in range(len(spClasedReduced[0][x0])):
                for x3 in range(len(spClasedReduced[0][x1])):
                    if isclose(spClasedReduced[0][x0][x2], spClasedReduced[0][x1][x3], rel_tol=1e-2)\
                    and isclose(spClasedReduced[1][x0][x2], spClasedReduced[1][x1][x3], rel_tol=1e-2)\
                    and isclose(spClasedReduced[2][x0][x2], spClasedReduced[2][x1][x3], rel_tol=1e-2):
                        same+=1
            if len(spClasedReduced[0][x0])>len(spClasedReduced[0][x1]):
                if same/len(spClasedReduced[0][x1])>0.5:
                    dels.append((x1,x0,x1,1))
            elif len(spClasedReduced[0][x0])<len(spClasedReduced[0][x1]):
                if same/len(spClasedReduced[0][x0])>0.5:
                    dels.append((x0,x0,x1,2))
            elif r1[rtheta_reduced[0][x0]][rtheta_reduced[1][x0]]>r1[rtheta_reduced[0][x1]][rtheta_reduced[1][x1]]:
                if same/len(spClasedReduced[0][x1])>0.5:
                    dels.append((x1,x0,x1,3))
            else:
                if same/len(spClasedReduced[0][x0])>0.5:
                    dels.append((x0,x0,x1,4))
    if len(dels)!=0:
        #print("Delsy:", dels)
        #print("Delsy:", np.transpose(dels)[0])
        unique_dels, _ = np.unique(np.transpose(dels)[0], axis=0, return_index=True)
        #print("Unique dels:", unique_dels)
        rtheta_reduced_same = np.delete(rtheta_reduced, unique_dels, 1)
        spClasedReduced_same = np.delete(spClasedReduced, unique_dels, 1)
        #print(rtheta_reduced_same, spClasedReduced_same)

        return rtheta_reduced_same, spClasedReduced_same
    else:
        return rtheta_reduced, spClasedReduced

def plotFinalResult(sp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(sp[0])):
        ax.scatter(sp[0][i], sp[1][i], sp[2][i], label=str(i), s=40, alpha=0.9)
    #ax.scatter(spx, spy, spz, c='skyblue', s=30, alpha=0.5)
    ax.set_ylabel("spy")
    ax.set_xlabel("spx")
    ax.set_zlabel("spz")
    ax.set_title("Fitted, labeled points")
    ax.view_init(30, 185)
    plt.title("Finalne maxima")
    plt.legend()
    plt.show()


def trackFindSmallNoise(entry, spx, spy, spz, spr):
    # mR,mT,mV, r1, rx, RXlims = loadInvTransParams(entry)
    mR, mT, mV, r1, rx, RXlims = HTG.inverseHoughTransformLines(spx, spy, spz, spr, entry,
                                                                RXlims=[1, 0, 1800, 1, -500, 500, 4.0, 10],
                                                                toFile=False, plot=False, plotMax=False)
    #print("\nMax. points:\t", len(mR), "\nR:\t\t\t", mR, "\ntheta:\t\t", mT, "\nValues:\t\t", mV,
    #      "\n{R,theta} ranges:\t", RXlims)

    spxClasd, spyClasd, spzClasd, rtheta_est_raw, r_est, theta_est = find_local_maxima(spx, spy, spz, r1, rx, RXlims,
                                                                                       plot=False, treshhold=0.0)
    if len(spxClasd)!=0:
        min_points_class = 4
        print("Found Maxima:", len(rtheta_est_raw[0]), "\nReducing maxima with less than", min_points_class, "points")
        plot3D(spx, spy, spz, spxClasd, spyClasd, spzClasd, title="Found raw maxima")
        # plot_found_maximum(spr, spz, r_est, theta_est)

        rtheta_reduced, spClasedReduced = ReduceMaximaLessThan_Points(rtheta_est_raw, spxClasd, spyClasd, spzClasd, rx,
                                                                      points=min_points_class)
        if len(rtheta_reduced[0])!=0:
            print("Found Maxima:", len(rtheta_reduced[0]), "\nReducing maxima with scattered points")
            plot3D(spx, spy, spz, spClasedReduced[0], spClasedReduced[1], spClasedReduced[2],
                   title="Reduced maxima with less than " + str(min_points_class) + " points")

            rtheta_reduced_same, spClasedReduced_same = ReduceMaximaXYscatteredByAngleDiffBySTD(rtheta_reduced, spClasedReduced, plot=False)
            if len(rtheta_reduced_same[0])!=0:
                print("Found Maxima:", len(rtheta_reduced_same[0]), "\nReducing maxima with the same points")
                plot3D(spx, spy, spz, spClasedReduced_same[0], spClasedReduced_same[1], spClasedReduced_same[2],
                       title="Reduced scattered XY points")

                rtheta, sp = ReduceMaximaWithSamePoints(rtheta_reduced_same, spClasedReduced_same, r1)
                if len(rtheta[0]) != 0:
                    print("Found Maxima:", len(rtheta[0]))
                    plot3D(spx, spy, spz, sp[0], sp[1], sp[2], title=str("Reducing maxima with overlapping points"))
                    plotFinalResult(sp)

                print(sp, rtheta, sep="\n\n")


## żle zrobione 680:0,

entries=np.array([302,311,316,324,177,353,594,103 ])#177,353,594])

#entries=np.array([256])
for i0 in range(782,790):
#for i0 in entries:
    entry = i0 #177, 353, 594, 595 i 982, 14 as a maximum amount of points(264) 103, 302, 311!, 316, 324, 333, 405!, 410
    spx, spy, spz, spr = HTG.loadEvent(entry)
    #HTG.plotEvent(spx, spy, spz, entry)
    if len(spx)<100000:
       trackFindSmallNoise(entry, spx, spy, spz, spr)


#nie łapie punktów ze środka i potem jest prosta 3 punkty a ja je odrzucam

#dobrać lepiej zmienne

#zmienić y0 na inną zmienną bo krzywdzi ona pionowe jety zaczynające się w (dalekie z, 0)
##można wrócić do koncepcji zmiennych orginalnej transformaty albo zastosować podwójną iterację z x0

#zwiększyć ilość dokładności promienia z 1024 na 10 000

#zmienić kryteria rozrzucenia z odległośći na kąt
#
print("\nfinit0")

