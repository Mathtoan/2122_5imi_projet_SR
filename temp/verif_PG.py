import sys
sys.path.insert(0, 'src')

from utils import *

H = np.loadtxt('HR_grid_0.txt')
save_im_new('test/HR_grid_0.png', H)
im_ref = rescale(rgb2gray(io.imread('fig/iPhone13Pro/scene1/IMG_1330.jpg')), 1/4)
im_sr, gauss_filter = PG_method(H, im_ref ,0.0625, 4, 300, out_filter=True, MSE=True, save_dir='test/2', intermediary_step=True)

io.imsave('test/2/test.png', im_sr.real)
plt.figure()
plt.imshow(im_sr.real, 'gray')
plt.colorbar()
plt.show()
plt.close()



# U,V,H = gaussian_2d(2268, 4032,0.125, 0.125, meshgrid=True)
# print(np.amin(H), np.amax(H))

# plt.figure()
# plt.imshow(H, 'gray')
# plt.colorbar()
# plt.show()
# plt.close()