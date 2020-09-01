import sys
import ROOT
import numpy as np



inFile = ROOT.TFile.Open("./data/measurements.root", "READ")
Tree = inFile.Get("SPTrkNtuple")

spxTemp=[]
spyTemp=[]
spzTemp=[]
spx=[]
spy=[]
spz=[]

"""
for event in Tree:
    if x<3:
        for i0 in range(len(event.spX)):
            spxTemp.append(np.add(event.spX[i0], 0))
            spyTemp.append(np.add(event.spY[i0], 0))
            spzTemp.append(np.add(event.spZ[i0], 0))
        spz.append(np.add(spzTemp, 0))
        spx.append(np.add(spxTemp, 0))
        spy.append(np.add(spyTemp, 0))
    x += 1

for event in Tree:
    if x<3:
        for i0 in range(len(event.spX)):
            spxTemp.append(str(event.spX[i0]))
            spyTemp.append(str(event.spY[i0]))
            spzTemp.append(str(event.spZ[i0]))
        spz.append(spzTemp)
        spx.append(spxTemp)
        spy.append(spyTemp)
    x += 1  
    
"""

pos="spy"
x1 = 0
f = open("./data/" + pos + str(x1 / 200), mode="w")
for event in Tree:
    if x1%200==0:
        f.close()
        f = open("./data/" + pos+str(x1/200), mode="w")
        print(x1)
    for i0 in range(len(event.spY)):
        f.write(str(event.spY[i0]))
    f.write("\n")
    x1 += 1
f.close()

def rootConverta(spx, pos="spx"):
    f = open("./data/"+pos, mode="w")
    for i in range(len(spx)):
        for i0 in range(len(spx[i])):
            f.write(spx[i][i0])
        f.write("\n")

print("Finito")