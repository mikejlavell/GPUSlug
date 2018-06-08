import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import interactive

N = 36
hmax=1.0
hmin=0.6

plt.figure(1)
x,y,temp = np.loadtxt('slug0Blast.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(True)
plt.colorbar()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Heat Map at Time $t=0$')
plt.show()


exit()

plt.figure(2)
x,y,temp = np.loadtxt('sol2048.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(True)
plt.colorbar()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Heat Map at Time $t=0.125$')
plt.show()


plt.figure(3)
x,y,temp = np.loadtxt('sol4096.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(True)
plt.colorbar()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Heat Map at Time $t=0.25$')
plt.show()






plt.figure(4)
x,y,temp = np.loadtxt('sol6144.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(True)
plt.xlabel('$X$')
plt.colorbar()
plt.ylabel('$Y$')
plt.title('Heat Map at Time $t=0.375$')
plt.show()


plt.figure(5)
x,y,temp = np.loadtxt('sol8192.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(False)
plt.colorbar()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Heat Map at Time $t=0.5$')
plt.show()

