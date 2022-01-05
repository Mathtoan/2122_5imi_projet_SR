import shutil
import time

import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import shift
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale


def display_progression(it, it_total):
    progression = (it/it_total) * 100
    if progression < 100:
        print('progression : %.2f%%' %(progression), end="\r")
    else:
        print('progression : %.2f%%' %(progression))
        print('done')

def creation_HR_grid(im_ref, upscale_factor, im_to_shift, shift, color):
    print('---- Creation HR grid ----')
    global_start_time = time.time()
    lr_size = im_ref.shape
    if color=='grey':
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
