import argparse
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from scipy.fftpack import fft2, ifft2

# Parser
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('-d', '--device', type=str, default='iPhone13Pro',
                    help='Choose the device', choices='iPhone13Pro')
parser.add_argument('-s', '--scene', type=str, default='scene1',
                    help='Choose the scene', choices='scene1')

args = parser.parse_args()

# Parameters
upscale_factor = 4
it = 100
sigma = 0.5

# Path
device = args.device
scene = args.scene
input_dir = os.path.join('fig', device, scene)
output_dir = 'output/up_'+str(upscale_factor)+'_it_'+str(it)+'_sigma_'+str(sigma)


list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()

def display_progression(it, it_total):
    progression = (it/it_total) * 100
    if progression < 100:
        print('progression : %.2f%%' %(progression), end="\r")
    else:
        print('progression : %.2f%%' %(progression))
        print('done')

def creation_HR_grid(im_ref, upscale_factor, im_to_shift, shift):
    print('---- Creation HR grid ----')
    global_start_time = time.time()
    lr_size = im_ref.shape
    sr_size = [lr_size[0]*upscale_factor, lr_size[1]*upscale_factor]
    im_sr = np.zeros(sr_size)

    for h in range(lr_size[0]):
        for w in range(lr_size[1]):
    
            idx_h_ref = h*upscale_factor+int(upscale_factor/2)
            idx_w_ref = w*upscale_factor+int(upscale_factor/2)

            im_sr[idx_h_ref][idx_w_ref] = im_ref[h][w]

            for k in range(len(shift)):
                idx_h = idx_h_ref + h*upscale_factor+shift[k][0]*upscale_factor
                idx_w = idx_w_ref + w*upscale_factor+shift[k][0]*upscale_factor

                if idx_h > 0 and idx_h < sr_size[0] and idx_w > 0 and idx_w < sr_size[1]:
                    im_sr[int(idx_h)][int(idx_w)] = im_to_shift[k][h][w]
            
            display_progression(h*lr_size[1]+w+1, lr_size[0]*lr_size[1])
    global_time = time.time() - global_start_time
    print('Execution time : %0.2fs' % (global_time))
    return im_sr

def PG_method(HR_grid, im_ref, sigma, upscale_factor, it):
    print('---- Papoulis–Gerchberg method ----')
    global_start_time = time.time()
    lr_size = im_ref.shape
    im_sr = HR_grid
    print(HR_grid.shape)
    for i in range(it):

        # fft_im_sr = fft2(im_sr)
        # print(fft_im_sr.shape)
        # fft_im_sr = gaussian(fft_im_sr, sigma)
        # im_sr = ifft2(fft_im_sr)
        im_sr = gaussian(im_sr, sigma)
        for h in range(lr_size[0]):
            for w in range(lr_size[1]):
                idx_h_ref = h*upscale_factor+int(upscale_factor/2)
                idx_w_ref = w*upscale_factor+int(upscale_factor/2)

                im_sr[idx_h_ref][idx_w_ref] = im_ref[h][w]
        
        display_progression(i, it)
    global_time = time.time() - global_start_time
    print('Execution time : %0.2f' % (global_time))
    return im_sr



# Registration and SR grid creation
# for idx_ref in range(len(list_image_input_dir)):
#     print('######### idx_ref =', idx_ref, '#########')
#     im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/8)
#     shift = []
#     to_shift = {}
#     count_val = 0
#     count_or = 0
#     for i in range(len(list_image_input_dir)):
#         if i != idx_ref:
#             im_to_register = rescale(rgb2gray(io.imread(list_image_input_dir[i])), 1/8)

#             shifted, error, diffphase = phase_cross_correlation(im_ref, im_to_register,upsample_factor=upscale_factor)
#             # registered_im = shift(im_to_register, shift=(shifted[0], shifted[1]), mode='constant')

#             # plt.figure()
#             # plt.subplot(221)
#             # plt.imshow(im_ref, 'gray')
#             # plt.title('Image de reference')
#             # plt.subplot(222)
#             # plt.imshow(im_to_register, 'gray')
#             # plt.title('Image a recaler')
#             # plt.subplot(223)
#             # plt.imshow(registered_im, 'gray')
#             # plt.title('Image recalee')
#             # plt.show()
#             # plt.close()
#             if shifted[0] != int(shifted[0]) or shifted[1] != int(shifted[1]):
#                 count_or += 1
#                 if not(shifted.tolist() in shift):
#                     shift.append(shifted.tolist())
#                     to_shift[list_image_input_dir[i]] = im_to_register
#                     count_val += 1
#             print(i, shifted)
#     print(shift)
#     print("at least 1 shift :", count_or, "| valid shifts :", count_val)

idx_ref = 9
print('######### idx_ref =', idx_ref, '#########')
im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/upscale_factor)
shift = []
to_shift = []
count_val = 0
count_or = 0
for i in range(len(list_image_input_dir)):
    if i != idx_ref:
        im_to_register = rescale(rgb2gray(io.imread(list_image_input_dir[i])), 1/upscale_factor)

        shifted, error, diffphase = phase_cross_correlation(im_ref, im_to_register,upsample_factor=upscale_factor)
        # registered_im = shift(im_to_register, shift=(shifted[0], shifted[1]), mode='constant')

        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(im_ref, 'gray')
        # plt.title('Image de reference')
        # plt.subplot(222)
        # plt.imshow(im_to_register, 'gray')
        # plt.title('Image a recaler')
        # plt.subplot(223)
        # plt.imshow(registered_im, 'gray')
        # plt.title('Image recalee')
        # plt.show()
        # plt.close()
        if shifted[0] != int(shifted[0]) or shifted[1] != int(shifted[1]):
            count_or += 1
            if not(shifted.tolist() in shift):
                shift.append(shifted.tolist())
                to_shift.append(im_to_register)
                count_val += 1
        print(i, shifted)
print(shift)
print("at least 1 shift :", count_or, "| valid shifts :", count_val)

HR_grid = creation_HR_grid(im_ref, upscale_factor, to_shift, shift)
im_sr = PG_method(HR_grid, im_ref, sigma, upscale_factor, it)

if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

io.imsave(os.path.join(output_dir,'groundtruth.png'), rgb2gray(io.imread(list_image_input_dir[0])))
io.imsave(os.path.join(output_dir,'lr_image.png'), im_ref)
io.imsave(os.path.join(output_dir,'hr_grid.png'), HR_grid)
io.imsave(os.path.join(output_dir,'sr_image.png'), im_sr)

plt.figure()
plt.subplot(221)
plt.imshow(rgb2gray(io.imread(list_image_input_dir[0])), 'gray')
plt.title('Groundtruth')
plt.subplot(222)
plt.imshow(im_ref, 'gray')
plt.title('LR image (im ref)')
plt.subplot(223)
plt.imshow(HR_grid, 'gray')
plt.title('HR grid')
plt.subplot(224)
plt.imshow(im_sr, 'gray')
plt.title('SR image')
plt.show()
plt.close()

