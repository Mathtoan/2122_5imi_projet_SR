import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from skimage import io
from mpl_toolkits import mplot3d
import os

def gaussian_kernel(l, sigma):
    ax = np.linspace(-(l-1)/2.,(l-1)/2., l)
    # ker_1d = np.exp(-(ax**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    ker_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    ker_2d = np.outer(ker_1d, ker_1d)
    return ker_2d/np.sum(ker_2d)

def gaussian_2d(h, w,sigma_x, sigma_y, meshgrid=False):
    # 'Normalized [-1,1] meshgrids' 
    u = np.linspace(-(w-1)/2.,(w-1)/2., w)/((w-1)/2)
    v = np.linspace(-(h-1)/2.,(h-1)/2., h)/((h-1)/2)
    U,V = np.meshgrid(u,v)

    H = np.exp(-0.5*((U/sigma_x)**2+(V/sigma_y)**2)) #/(np.sqrt(2*np.pi*sigma_x*sigma_y))
    if not(meshgrid):
        return H
    else:
        return U, V, H

h,w = (2268, 4032)
sigma_x,sigma_y =100, 100

U,V,H = gaussian_2d(h, w,sigma_x,sigma_y,meshgrid=True)
# print(np.amax(H), np.amin(H))
# io.imsave(os.path.join('temp', 'H.png'), H)

# plt.figure()
# plt.imshow(H, cmap='gray')
# plt.colorbar()
# plt.show()
# plt.close()

HR_grid = np.loadtxt('output/iPhone13Pro/scene1/up_4/HR_grid_9.txt')
HR_grid_fft = abs(fftshift(fft2(HR_grid)))

plt.figure()
plt.imshow(np.log10(HR_grid_fft), 'gray')
plt.show()
plt.close()

max_val = np.amax(HR_grid_fft)
min_val = np.amin(HR_grid_fft)



print(max_val, min_val)

HR_grid_fft_norm = (HR_grid_fft-min_val)/(max_val-min_val)
print(U[::4,::4].shape,V[::4,::4].shape,HR_grid_fft_norm[::4,::4].shape)
print(np.amax(HR_grid_fft_norm), np.amin(HR_grid_fft_norm))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(U[::4,::4], V[::4,::4], HR_grid_fft_norm[::4,::4], c=HR_grid_fft_norm[::4,::4], cmap='Greens')
# plt.show()