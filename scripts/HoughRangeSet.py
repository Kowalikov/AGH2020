import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys

def plotError():
    x=np.linspace(6,100,91)
    y=(x-np.sqrt(np.multiply(x,x)-36))/6
    plt.plot(x,y)
    plt.ylim(0, y.max()*1.1)
    plt.grid()
    plt.ylabel("Znormalizowana rozbieżność")
    plt.xlabel("Promień okręgu")
    plt.title("Rozbieżność końców fragmentu okręgu i prostej na granicy detektora ")
    plt.show()

def plotMSE(dlim=6, uplim=100, plot=False):
    mse=[]
    for a in range (dlim,uplim):
        x = np.linspace(0, 6, 61)
        yest = np.zeros(61)
        y = (a - np.sqrt(a**2-np.multiply(x, x)))
        if plot==True:
            plt.plot(x, y)
            plt.grid()
            plt.title("R="+str(a)+" MSE="+str((np.square(y - x)).mean(axis=None)))
            plt.show()
        mse.append((np.square(yest - y)).mean(axis=0))
    a=np.linspace(dlim,uplim-1, (uplim-dlim))
    print(a)
    plt.plot(a, mse)
    plt.ylabel("MSE")
    plt.xlabel("Promień okręgu")
    plt.title("Błąd średniokwadratowy od promienia")
    plt.ylim(0,0.2)
    plt.grid()
    plt.show()
    return(mse)

#plotError()
plotMSE()
print("finito")
