import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def gaussian_kernel(l, sigma):
    ax = np.linspace(-(l-1)/2.,(l-1)/2., l)
    # ker_1d = np.exp(-(ax**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    ker_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    ker_2d = np.outer(ker_1d, ker_1d)
    return ker_2d/np.sum(ker_2d)

def gaussian_2d(h, w,sigma_x, sigma_y, meshgrid=False):
    u = np.linspace(-(w-1)/2.,(w-1)/2., w)/((w-1)/2)
    v = np.linspace(-(h-1)/2.,(h-1)/2., h)/((h-1)/2)
    U,V = np.meshgrid(u,v)
    H = np.exp(-0.5*((U/sigma_x)**2+(V/sigma_y)**2))/(sigma_x*sigma_y*np.sqrt(2*np.pi))
    if not(meshgrid):
        return H
    else:
        return U, V, H

h,w = (2268, 4032)
sigma_x,sigma_y = 100, 100*h/w

U,V,H = gaussian_2d(h, w,sigma_x,sigma_y, meshgrid=True)

plt.figure()
plt.imshow(H, cmap='gray')
plt.show()
plt.close()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(U, V, X, c=X, cmap='Greens')
# plt.show()