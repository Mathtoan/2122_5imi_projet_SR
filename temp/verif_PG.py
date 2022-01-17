import sys
sys.path.insert(0, 'src')

from utils import *


HR_grid = np.loadtxt('debug/iPhone13Pro/scene1/psf/up_4/HR_grid_0.txt')

hist, bin = np.histogram(HR_grid, bins=np.linspace(1/255,1,256))
print(hist, bin)

plt.figure()
plt.hist(HR_grid.reshape((HR_grid.shape[0]*HR_grid.shape[1],)), bins=np.linspace(1/255,1,256), )
plt.title('Histogram HR Grid without 0')
plt.grid(True, axis='y')
plt.savefig('test/9/hist_HR_grid_1.png', dpi=200)
plt.close()
# save_im('test/HR_grid_0.png', HR_grid)
im_ref = rgb2gray(io.imread('fig/iPhone13Pro/scene1/IMG_1330.jpg'))
plt.figure()
plt.hist(im_ref.reshape((im_ref.shape[0]*im_ref.shape[1],)), bins=np.linspace(0,1,256), )
plt.title('Histogram groundtruth')
plt.grid(True, axis='y')
plt.savefig('test/9/hist_gt.png', dpi=200)
plt.close()

im_sr = np.loadtxt('/Users/Toan/5ETI/2122_5imi_projet_SR/test/9/it_10/sr_image.txt')
plt.figure()
plt.hist(im_sr.reshape((im_sr.shape[0]*im_sr.shape[1],)), bins=np.linspace(0,1,256), )
plt.title('Histogram SR image')
plt.grid(True, axis='y')
plt.savefig('test/9/hist_im_sr.png', dpi=200)
plt.close()

# for i in range(10):
#     im_sr, filter = PG_method(HR_grid, im_ref, 0.25, 4, 10,
#                                     out_filter=True, intermediary_step=False, save_dir='test/9/', MSE=True, psf=False,
#                                     imshow_debug=False, plot_debug_intensity=True)

# io.imsave('test/9/test.png', im_sr.real)
# plt.figure()
# plt.imshow(HR_grid, 'gray')
# plt.colorbar()
# plt.show()
# plt.close()



# U,V,H = gaussian_2d(2268, 4032,0.125, 0.125, meshgrid=True)
# print(np.amin(H), np.amax(H))

# plt.figure()
# plt.imshow(H, 'gray')
# plt.colorbar()
# plt.show()
# plt.close()