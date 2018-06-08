import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

# get data
data0 = np.genfromtxt('slug0Blast.dat')

print(len(data0))

x=data0[:,0]
y=data0[:,1]
temp_t00=data0[:,2]

# define mesh
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# interpolate surfaces
rbf0 = scipy.interpolate.Rbf(x, y, temp_t00, function='linear')
zi0 = rbf(xi, yi)


#surface plot t=0
plt.imshow(zi0, vmin=temp_t00.min(), vmax=temp_t00.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.scatter(x, y, c=temp_t00)
plt.colorbar()
plt.title('Temperature Distribution $t=0$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



#surface plot t=0.5
#plt.plot(data1[:,0],data1[:,1],'g',label="t=0.5")


#surface plot t=1.0
#plt.plot(data2[:,0],data2[:,1],'b',label="t=1.0")


#surface plot t=1.5
#plt.plot(data3[:,0],data3[:,1],'k',label="t=1.5")

