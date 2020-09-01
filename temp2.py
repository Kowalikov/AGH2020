import numpy as np
import matplotlib.pyplot as plt


ran=np.linspace(1,5,20)
#angle = np.random.uniform(0, 2*np.pi, size=(10))
phi0=np.pi/4
angle = np.linspace(0+phi0, np.pi/4+phi0,20)
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


ran1=np.where(ran>40)
#ran=ran[ran1]
#angle = angle[ran1]

print(len(ran1[0]), ran1)

"""fig, (ax1, ax2) = plt.subplots(1, 2)

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
plt.show()"""



"""
a=5
b=0
r=5

x=np.linspace(0.0,10,1000)
#print(x)
y=np.sqrt(r**2-np.multiply(x-a,x-a)+b)


R=np.sqrt(np.square(x)+np.square(y))
"""