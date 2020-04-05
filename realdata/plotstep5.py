import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


data = np.load('GDELTstep5(440.3).npy')
print(np.where(data == np.min(data[:,3])))


R = data[:,3]

fig = plt.figure(figsize = (10,7))
ax=Axes3D(fig)
X = np.array([-2,-1,0,1,2])
Y = np.array([-2,-1,0,1,2])
X, Y = np.meshgrid(X, Y)

xticks = [0.01,0.1,1,10,100]
yticks = [0.01,0.1,1,10,100]
ax.set_xlabel(r'$ \alpha $-time')
ax.set_ylabel(r'$ \alpha $-event')
ax.set_zlabel('Relative error')
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
ax.set_yticks(np.log10(yticks))
ax.set_yticklabels(yticks)

# Customize the z axis.
# ax.set_zlim(0.209, 0.212)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, shade=True)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(15,45)
plt.show()
fig.savefig('plotstep5.png',dpi=300)