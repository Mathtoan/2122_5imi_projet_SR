import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft2, ifft2
from scipy.ndimage import shift
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale


def display_progression(it, it_total):
    progression = (it/it_total) * 100
    progress_bar = '['+'-'*(int(progression)//5)+' '*(int(100/5)-int(progression)//5)+']'
    if progression < 100:
        print('Loading '+progress_bar+' %.2f%%' %(progression), end="\r")
    else:
        print('Loading '+progress_bar+' %.2f%%' %(progression))
        print('done')

def computing_regitration(list_image_input_dir, idx_ref, upscale_factor, display=False):
    print('######### idx_ref =', idx_ref, '#########')
    im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/upscale_factor)
    registration_shifts = []
    im_to_register_list = []
    count_val = 0
    count_or = 0
    for i in range(len(list_image_input_dir)):
        if i != idx_ref:
            im_to_register = rescale(rgb2gray(io.imread(list_image_input_dir[i])), 1/upscale_factor)

            shifted, _, _ = phase_cross_correlation(im_ref, im_to_register,upsample_factor=upscale_factor)

            if display:
                registered_im = shift(im_to_register, shift=(shifted[0], shifted[1]), mode='constant')
                plt.figure()
                plt.subplot(221)
                plt.imshow(im_ref, 'gray')
                plt.title('Image de reference')
                plt.subplot(222)
                plt.imshow(im_to_register, 'gray')
                plt.title('Image a recaler')
                plt.subplot(223)
                plt.imshow(registered_im, 'gray')
                plt.title('Image recalee')
                plt.show()
                plt.close()

            if shifted[0] != int(shifted[0]) or shifted[1] != int(shifted[1]):
                if not(shifted.tolist() in registration_shifts):
                    registration_shifts.append(shifted.tolist())
                    im_to_register_list.append(im_to_register)
                    count_val += 1
            print(i, shifted)
    print(registration_shifts)
    print("valid shifts :", len(im_to_register_list))
    return im_ref, im_to_register_list, registration_shifts

def creation_HR_grid(im_ref, upscale_factor, im_to_register_list, registration_shifts, color):
    print('---- Creation HR grid ----')
    global_start_time = time.time()
    lr_size = im_ref.shape
    if color=='gray':
        sr_size = [lr_size[0]*upscale_factor, lr_size[1]*upscale_factor]
    elif color=='rgb':
        sr_size = [lr_size[0]*upscale_factor, lr_size[1]*upscale_factor, 3]
    else:
        print('Undefined color')
        exit()
    im_sr = np.zeros(sr_size)

    for h in range(lr_size[0]):
        for w in range(lr_size[1]):
    
            idx_h_ref = h*upscale_factor+int(upscale_factor/2)
            idx_w_ref = w*upscale_factor+int(upscale_factor/2)

            im_sr[idx_h_ref][idx_w_ref] = im_ref[h][w]

            for k in range(len(registration_shifts)):
                idx_h = idx_h_ref + h*upscale_factor+registration_shifts[k][0]*upscale_factor
                idx_w = idx_w_ref + w*upscale_factor+registration_shifts[k][0]*upscale_factor

                if idx_h > 0 and idx_h < sr_size[0] and idx_w > 0 and idx_w < sr_size[1]:
                    im_sr[int(idx_h)][int(idx_w)] = im_to_register_list[k][h][w]
            
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
        
        display_progression(i+1, it)
    global_time = time.time() - global_start_time
    print('Execution time : %0.2f' % (global_time))
    return im_sr
