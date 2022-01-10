import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import shift
from scipy.signal import convolve2d
from skimage import img_as_ubyte
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from tqdm import tqdm


def display_progression(it, it_total): # Old, should use tqdm instead
    progression = (it/it_total) * 100
    progress_bar = '['+'-'*(int(progression)//5)+' '*(int(100/5)-int(progression)//5)+']'
    if progression < 100:
        print('Loading '+progress_bar+' %.2f%%' %(progression), end="\r")
    else:
        print('Loading '+progress_bar+' %.2f%%' %(progression))
        print('done')
def save_im_new(path, im):
    if not(os.path.exists(path)):
        print('saving', path)
        io.imsave(path, im)
    else:
        print(path, 'already saved')
# psf2otf
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

def computing_regitration(list_image_input_dir, idx_ref, upscale_factor, display=False):
    print('######### idx_ref =', idx_ref, '#########')
    im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/upscale_factor)
    registration_shifts = []
    im_to_register_list = []
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
            print(i, shifted)
    print(registration_shifts)
    print("valid shifts :", len(im_to_register_list))
    return im_ref, im_to_register_list, registration_shifts

def computing_regitration_v2(im_ref, list_image_input_dir, idx_ref, upscale_factor, color, display=False):
    print('######### idx_ref =', idx_ref, '#########')
    im_registered_list = []
    for i in range(len(list_image_input_dir)):
        if i != idx_ref:
            im_to_register = io.imread(list_image_input_dir[i])
            if color=='gray':
                im_to_register = rgb2gray(im_to_register)
            # plt.imsave('rescaled.png',im_to_register)
            im_to_register = rescale(im_to_register, 1/upscale_factor)
            # im_to_register = rescale(im_to_register, upscale_factor)
            # print(im_ref.shape, im_to_register.shape)
            im_to_register_cv = img_as_ubyte(im_to_register)
            im_ref_cv = img_as_ubyte(im_ref)

            height, width = im_ref.shape
            orb_detector = cv2.ORB_create(5000)
            kp_recal, d_recal = orb_detector.detectAndCompute(im_to_register_cv, None)
            kp_ref, d_ref = orb_detector.detectAndCompute(im_ref_cv, None)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
            matches = matcher.match(d_recal, d_ref)
            matches = sorted(matches,key = lambda x: x.distance)
            matches = matches[:int(len(matches)*0.9)]
            no_of_matches = len(matches)
            # print(no_of_matches)
            p1 = np.zeros((no_of_matches, 2))
            p2 = np.zeros((no_of_matches, 2))
            for i in range(len(matches)):
                p1[i, :] = kp_recal[matches[i].queryIdx].pt
                p2[i, :] = kp_ref[matches[i].trainIdx].pt
            homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)

            im_to_register = rescale(im_to_register, upscale_factor)
            registered_im = cv2.warpPerspective(im_to_register, homography, (width*upscale_factor, height*upscale_factor))
            # print(registered_im.shape)

            if display:
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

            im_registered_list.append(registered_im)
            # plt.imsave('registered.png', registered_im)
            # exit()
    return im_registered_list

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

    for h in tqdm(range(lr_size[0]), desc='Main loop'):
        for w in tqdm(range(lr_size[1]), desc=f'Line {h}', leave=False):
    
            idx_h_ref = h*upscale_factor#+int(upscale_factor/2)
            idx_w_ref = w*upscale_factor#+int(upscale_factor/2)

            im_sr[idx_h_ref][idx_w_ref] = im_ref[h][w]

            for k in range(len(registration_shifts)):
                idx_h = idx_h_ref + registration_shifts[k][0]*upscale_factor
                idx_w = idx_w_ref + registration_shifts[k][0]*upscale_factor

                if idx_h > 0 and idx_h < sr_size[0] and idx_w > 0 and idx_w < sr_size[1]:
                    im_sr[int(idx_h)][int(idx_w)] = im_to_register_list[k][h][w]
            
    global_time = time.time() - global_start_time
    print('Execution time : %0.2fs' % (global_time))
    return im_sr

def creation_HR_grid_v2(im_ref, upscale_factor, im_registered_list, color):
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
    print("sr size = ", sr_size)

    for h in tqdm(range(lr_size[0]), desc='Main loop'):
        for w in tqdm(range(lr_size[1]), desc=f'Line {h}', leave=False):
    
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor

            im_sr[idx_h_ref][idx_w_ref] = im_ref[h][w]

    for k in range(len(im_registered_list)):
        im_sr += im_registered_list[k]
            
    global_time = time.time() - global_start_time
    print('Execution time : %0.2fs' % (global_time))
    return im_sr

def PG_method(HR_grid, im_ref, sigma, upscale_factor, it, out_filter=False, save_every=False, save_dir=None):
    if save_every:
        if save_dir==None:
            print('A save path should be input')
            exit()
    print('---- Papoulis–Gerchberg method (real)----')
    # global_start_time = time.time()
    lr_size = im_ref.shape
    im_sr = HR_grid
    sr_size = HR_grid.shape
    print(HR_grid.shape)
    gauss_filter = gaussian_2d(sr_size[0], sr_size[1], sigma, sigma)
    for i in tqdm(range(it), desc='Main loop'):

        fft_im_sr = fftshift(fft2(im_sr))
        fft_im_sr = fft_im_sr * gauss_filter
        im_sr = ifft2(ifftshift(fft_im_sr))

        im_sr[HR_grid>0]  = HR_grid[HR_grid>0]
        
        if save_every and (i+1)%100 == 0:
            save_path = os.path.join(save_dir, 'it_'+str(i+1))
            if not(os.path.exists(save_path)):
                os.makedirs(save_path)
            save_im_new(os.path.join(save_path,'sr_image_new.png'), im_sr.real)
        
    # global_time = time.time() - global_start_time
    # print('Execution time : %0.2f' % (global_time))
    if not(out_filter):
        return im_sr
    else:
        return im_sr, gauss_filter
