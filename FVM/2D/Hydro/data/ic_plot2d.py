import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import interactive

N = 132
hmax=11.0
hmin=0.0

plt.figure(1)
x,y,temp = np.loadtxt('slug0Blast.dat').T #Transposed for easier unpacking
nrows, ncols = N, N
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, vmin=hmin, vmax=hmax, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest')
interactive(False)
plt.colorbar()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Initial pressure')
#plt.show()


print(max(temp))
print(len(x))
print(x)
print(len(y))
print(y)

plt.show()
