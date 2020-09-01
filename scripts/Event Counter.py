import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import sys

spxTemp= []
entry = 177
entryLen = 0
rangeRead=12
acc=0
for x in range(rangeRead):
    f = open("../data/raw_data/spy"+str(x)+".0")
    x0=0
    for line in f:
        spxTemp.append(line.split(" "))
        i0=len(spxTemp[x0][:-2])
        print(i0)
        if entry==(x0+x*200):
            entryLen=i0
        acc+=len(spxTemp[x0][:-2])
        x0+=1


print("Acc: ", acc)
print("Entry ", entry, " len: ", entryLen)


print("finito")
