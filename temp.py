import numpy as np
import matplotlib.pyplot as plt


ran=np.linspace(1,5,100)
#angle = np.random.uniform(0, 2*np.pi, size=(10))
phi0=np.pi/4
angle = np.linspace(0+phi0, 4*np.pi/4+phi0,100)
angle[[3,5]]=np.random.uniform(0, 2*np.pi, size=(2))

print(angle, ran)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(np.multiply(np.cos(angle),ran), np.multiply(np.sin(angle),ran), "+")
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.grid()
plt.grid()

ax2.set_xlim(left=0, right=10)
ax2.set_ylim(0, 360)
ax2.grid()
ax2.plot(ran, angle * 180 / np.pi, '+')
#ax2.plot(distXY, (a * (distXY ** 3) + m * (distXY ** 2) + b * distXY + c) * 180 / np.pi, '-')
plt.grid()
plt.show()

a=5
b=0
r=5

x=np.linspace(0.0,10,1000)
#print(x)
y=np.sqrt(r**2-np.multiply(x-a,x-a)+b)


R=np.sqrt(np.square(x)+np.square(y))

"""
plt.plot(x,y,'+')
plt.grid()
plt.show()
plt.plot(x,R,'+')
plt.grid()
plt.show()"""

phi=np.zeros(shape=(len(x)))
print(len(x))
for x0 in range(len(x)):
    if x[x0] == 0:
        if y[x0] > 0:
            phi[x0]=np.pi / 2
        else:
            phi[x0]=3 * np.pi / 2
    elif x[x0] < 0:
        phi[x0]=np.arctan(y[x0] / x[x0]) + np.pi
    elif y[x0] < 0:
        phi[x0]=np.arctan(y[x0] / x[x0]) + 2 * np.pi
    else:
        phi[x0]=np.arctan(y[x0] / x[x0])

#print(phi)

"""plt.plot(R,phi*180/np.pi,'+')
plt.ylim(0,90)
plt.grid()
plt.show()

m,b = np.polyfit(R[2:300], phi[2:300], 1)

plt.plot(R,phi*180/np.pi,'+')
plt.plot(R,(m*R+b)*180/np.pi,'-')
plt.plot(R,(m*R+b-phi)*180/np.pi,'-')
plt.ylim(0,90)
plt.grid()
plt.show()

m,b,c = np.polyfit(R[2:1000], phi[2:1000], 2)

plt.plot(R,phi*180/np.pi,'+')
plt.plot(R,(m*R**2+b*R+c)*180/np.pi,'-')
plt.ylim(0,90)
plt.grid()
plt.show()

a,m,b,c = np.polyfit(R[1:1000], phi[1:1000], 3)

plt.plot(R,phi*180/np.pi,'+')
plt.plot(R,(a*(R**3)+m*(R**2)+b*R+c)*180/np.pi,'-')
plt.ylim(0,90)
plt.grid()
plt.show()
"""