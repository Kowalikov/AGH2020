import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys

PI = 3.14159



cx=[]
cy=[]
r0=3
a0=1
b0=-2
#add circle with noise
for t in range(0,90,5):
    cxTemp=a0+r0 * np.cos(t * PI / 180)+np.random.random_integers(-100,100,size=1)/500
    cyTemp=b0+r0 * np.sin(t * PI / 180)+np.random.random_integers(-100,100,size=1)/500
    print(cxTemp, cyTemp)
    cx.append(cxTemp[0])
    cy.append(cyTemp[0])
#add noise
for t in range(0,50):
    cxTemp=a0+np.random.random_integers(-600,600,size=1)/100
    cyTemp=b0+np.random.random_integers(-600,600,size=1)/100
    print(cxTemp, cyTemp)
    cx.append(cxTemp[0])
    cy.append(cyTemp[0])


plt.plot(cx, cy, "+")
plt.show()


size= len(cx)

Aa = []
Ab = []
Ar = []
rlim=5
for r in range(0,rlim):
    for x1 in range(size):
        for t in range(0,360,20):
                a = cx[x1]- r * np.cos(t * PI / 180)
                b = cy[x1] - r * np.sin(t * PI / 180)
                Aa.append(a)
                Ab.append(b)
                Ar.append(r)

print(Aa, Ab, Ar, sep="\n")
st = int(len(Aa)/rlim)
for x0 in range(rlim):
    xt=Aa[x0*st:((x0+1)*st-1)]
    yt=Ab[x0*st:((x0+1)*st-1)]
    plt.plot(xt, yt, "+")
    plt.title("R="+str(x0))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True,
                            tight_layout=True)
    axs.hist2d(xt, yt, bins=(30, 30), norm=colors.LogNorm())
    axs.grid()
    tit = "R="+str(x0)
    axs.set_title(tit)
    plt.show()



print("finito")
