import numpy as np


rtheta_est_raw = np.array([[1,2,4], [1,3,9]]) #tablica r-ów i thet, żeby znaleźć dla nich należące punkty (ich indeksy)
rx= np.arange(0,1000).reshape(10,10,10) #tablica indeksów punktów należących do danych r-ów i thet
spx=np.linspace(0,100, 1001) #tabliza położeń punktów
spy=np.linspace(0,1000,1001)
spz=np.linspace(-100,0,1001)
print(rx, spx, spy, spz, sep="\n")

for i in range(len(rtheta_est_raw[0])):
    print(rx[rtheta_est_raw[0][i]][rtheta_est_raw[1][i]])
    temp=np.array(rx[rtheta_est_raw[0][i]][rtheta_est_raw[1][i]])
    for counter, value in enumerate(temp):
        print(spx[value], end="\t")


#print(type(fit_points), fit_points[0])
