import numpy as np
import matplotlib.pyplot as plt
import os

i=1
ga = 2.0
var = ['position','density','velocity','pressure','eint']
m0 = []

data=np.loadtxt('slug1Sod.dat')

plt.plot(data[0,:],data[1,:],'-b',label='density')
plt.plot(data[0,:],data[2,:],'-r',label='velocity')
plt.plot(data[0,:],data[3,:],'-g',label='pressure')



plt.title('Sod Shock Tube')
#plt.ylabel(var[v])
plt.xlabel('position')
#plt.ylabel('pressure')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()


















