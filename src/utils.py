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

def float64_to_uint8(x):
    if x.dtype == np.uint8:
        return(x)
    elif x.dtype != np.float64:
        print('Data not in float64')
        exit()
    else:
        return np.round(x*255).astype(np.uint8)

def save_im(path, im, new=False):
    if os.path.exists(path) and new:
        print(path, 'already saved')
    else:
        io.imsave(path, float64_to_uint8(im))
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

def filter_test(h, w,sigma):
    u = np.linspace(-(w-1)/2.,(w-1)/2., w)/((w-1)/2)
    v = np.linspace(-(h-1)/2.,(h-1)/2., h)/((h-1)/2)
    U,V = np.meshgrid(u,v)

    H = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if abs(U[i][j])<sigma and abs(V[i][j])<sigma:
                H[i][j] = 1.
    return H

def creation_HR_grid(im_ref, list_image_input_dir, idx_ref, upscale_factor, color):
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
    # im_sr = np.zeros(sr_size)
    im_ref_up = np.zeros(sr_size)
    print("sr size = ", sr_size)

    for h in range(lr_size[0]):
        for w in range(lr_size[1]):
    
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor

            im_ref_up[idx_h_ref][idx_w_ref] = im_ref[h][w]
    im_sr = np.copy(im_ref_up)

    # print(np.amin(im_sr), np.amax(im_sr))
    print('######### idx_ref =', idx_ref, '#########')
    for k in tqdm(range(len(list_image_input_dir)), desc='Registration'):
        if k != idx_ref:
            im_to_register = io.imread(list_image_input_dir[k])
            if color=='gray':
                im_to_register = rgb2gray(im_to_register)
            im_to_register = rescale(im_to_register, 1/upscale_factor)
            registered_im = computing_regitration_POI(im_ref, im_to_register, upscale_factor)

            # im_sr[im_sr==0] = registered_im[im_sr==0]
            im_sr += registered_im
            
            plt.imsave("registered_"+str(k)+".png",im_ref_up+registered_im, cmap='gray')
            
    print("avant",np.amin(im_sr), np.amax(im_sr))
    im_sr = (im_sr - np.amin(im_sr)) / (np.amax(im_sr) - np.amin(im_sr))
    print("après",np.amin(im_sr), np.amax(im_sr))
            
    global_time = time.time() - global_start_time
    print('Execution time : %0.2fs' % (global_time))
    return im_sr

def computing_regitration_translation(im_ref, im_to_register, upscale_factor, display=False):    
    shifted, _, _ = phase_cross_correlation(im_ref, im_to_register,upsample_factor=upscale_factor)
    
    lr_size = im_ref.shape
    sr_size = [lr_size[0]*upscale_factor,lr_size[1]*upscale_factor]
    registered_im = np.zeros(sr_size)
            
    for h in range(lr_size[0]):
        for w in range(lr_size[1]):
            idx_h = h*upscale_factor + shifted[0]*upscale_factor
            idx_w = w*upscale_factor + shifted[1]*upscale_factor
            if idx_h > 0 and idx_h < sr_size[0] and idx_w > 0 and idx_w < sr_size[1]:
                registered_im[int(idx_h)][int(idx_w)] = im_to_register[h][w]

    if display:
        # registered_im = shift(im_to_register, shift=(shifted[0], shifted[1]), mode='constant')
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
    
    return registered_im

def computing_regitration_POI(im_ref, im_to_register, upscale_factor, display=False):
    
    height, width = im_ref.shape
    im_to_register_cv = img_as_ubyte(im_to_register)
    im_ref_cv = img_as_ubyte(im_ref)

    orb_detector = cv2.ORB_create(5000)
    kp_recal, d_recal = orb_detector.detectAndCompute(im_to_register_cv, None)
    kp_ref, d_ref = orb_detector.detectAndCompute(im_ref_cv, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(d_recal, d_ref)
    matches = sorted(matches,key = lambda x: x.distance)
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp_recal[matches[i].queryIdx].pt
        p2[i, :] = kp_ref[matches[i].trainIdx].pt
    homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)


    homography = np.linalg.inv(homography)
    registered_im = np.zeros([height*upscale_factor,width*upscale_factor])
    for h in range(height):
        for w in range(width):
            [x_prime, y_prime, s] = np.matmul(homography,[h,w,1])/np.matmul(homography[2],[h,w,1])
            idx_h_ref = int(x_prime*upscale_factor)
            idx_w_ref = int(y_prime*upscale_factor)
            if idx_h_ref>0 and idx_h_ref<height*upscale_factor and idx_w_ref>0 and idx_w_ref<width*upscale_factor:
                registered_im[idx_h_ref][idx_w_ref] = im_to_register[h][w]
    
    # im_to_register_up = np.zeros([height*upscale_factor,width*upscale_factor])
    # for h in range(height):
    #     for w in range(width):
    #         idx_h_ref = h*upscale_factor
    #         idx_w_ref = w*upscale_factor
    #         im_to_register_up[idx_h_ref][idx_w_ref] = im_to_register[h][w]     
       
    # registered_im = cv2.warpPerspective(im_to_register_up, homography, (width*upscale_factor, height*upscale_factor), cv2.INTER_NEAREST)
    # registered_im = cv2.warpPerspective(im_to_register, homography, (width, height))
    
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

    return registered_im

def computing_regitration_pixel(im_ref, im_to_register, upscale_factor, display=False):
    
    height, width = im_ref.shape
    im_to_register_cv = img_as_ubyte(im_to_register)
    im_ref_cv = img_as_ubyte(im_ref)
    
    im_to_register_cv = im_to_register.reshape((1,-1,2)).astype('float32')
    im_ref_cv = im_ref.reshape((1,-1,2)).astype('float32')
    
    homography, _ = cv2.findHomography(im_ref_cv, im_to_register_cv, method=0)
    # homography = np.linalg.inv(homography)
    # # print(homography)
    # height_cv, _, width_cv = im_to_register_cv.shape
    # print(im_to_register_cv.shape)
    # registered_im = np.zeros([height_cv, width_cv])
    # for i in range(height_cv):
    #     for j in range(width_cv):
    #         [x_prime, y_prime, s] = np.matmul(homography,[i,j,1])/np.matmul(homography[2],[i,j,1])
    #         x_prime = int(np.floor(x_prime))
    #         y_prime = int(np.floor(y_prime))
    #         if x_prime>0 and x_prime<height_cv and y_prime>0 and y_prime<width_cv:
    #             registered_im[x_prime,y_prime] = im_to_register_cv[i,0,j]
    #             # recal[i,j] = im_recal[x_prime,y_prime]


    # homography = np.linalg.inv(homography)
    # registered_im = np.zeros([height*upscale_factor,width*upscale_factor])
    # for h in range(height):
    #     for w in range(width):
    #         [x_prime, y_prime, s] = np.matmul(homography,[h,w,1])/np.matmul(homography[2],[h,w,1])
    #         idx_h_ref = int(x_prime*upscale_factor)
    #         idx_w_ref = int(y_prime*upscale_factor)
    #         if idx_h_ref>0 and idx_h_ref<height*upscale_factor and idx_w_ref>0 and idx_w_ref<width*upscale_factor:
    #             registered_im[idx_h_ref][idx_w_ref] = im_to_register[h][w]
    
    # registered_im = registered_im.reshape((height, width))
    
    
    
    im_to_register_up = np.zeros([height*upscale_factor,width*upscale_factor])
    for h in range(height):
        for w in range(width):
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor
            im_to_register_up[idx_h_ref][idx_w_ref] = im_to_register[h][w]
            
    im_to_register_up_cv = img_as_ubyte(im_to_register_up)
           
    registered_im = cv2.warpPerspective(im_to_register_up_cv, homography, (width*upscale_factor, height*upscale_factor), cv2.INTER_NEAREST)
    # registered_im = cv2.warpPerspective(im_to_register, homography, (width, height))
    
    registered_im = registered_im.reshape((height*upscale_factor, width*upscale_factor))
    
    plt.imsave('registered.png', registered_im, cmap='gray')
    exit()
    
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

    return registered_im


def PG_method(HR_grid, im_ref, sigma, upscale_factor, it,
              out_filter=False, intermediary_step=False, save_dir=None, MSE=None, psf=False,
              imshow_debug=False, plot_debug_idx=None):
    if intermediary_step:
        if save_dir==None:
            print("A save path should be input : won't save intermediary steps")
            intermediary_step = False
    print('---- Papoulis–Gerchberg method (real)----')
    # global_start_time = time.time()
    lr_size = im_ref.shape
    im_sr = HR_grid
    sr_size = HR_grid.shape
    print(HR_grid.shape)
    filter = gaussian_2d(sr_size[0], sr_size[1], sigma, sigma)
    # filter[filter<1e-5] = 0.
    # print("/!\ : in filter, forcing 0 when <1e-5")
    if MSE!=None:
        err = np.zeros(it)
    if plot_debug_idx!=None:
        intensity = np.zeros(it+1)
    # filter = filter_test(sr_size[0], sr_size[1], sigma)
    for i in tqdm(range(it), desc='Main loop'):
        old_im_sr = im_sr

        if not(psf):
            old_fft_im_sr = fftshift(fft2(old_im_sr))

            if i==0 and save_dir!=None:
                save_im(os.path.join(save_dir, 'fft.png'), np.log10(np.abs(old_fft_im_sr)))
                
            # fft_im_sr = old_fft_im_sr * filter
            fft_im_sr = np.multiply(old_fft_im_sr, filter)

            im_sr = ifft2(ifftshift(fft_im_sr))
        
        else:
            im_sr = gaussian(old_im_sr, sigma)

        # Plotting for debug purposes
        if imshow_debug and not(psf):
            # image and FFT between 2 consecutive interations
            plt.figure()
            plt.subplot(221)
            plt.imshow(old_im_sr.real, 'gray')
            plt.title('Image (it = %i)'%(i))

            plt.subplot(222)
            plt.imshow(np.log10(np.abs(old_fft_im_sr)), 'gray')
            plt.title('FFT image (log10) (it = %i)'%(i))

            plt.subplot(223)
            plt.imshow(im_sr.real, 'gray')
            plt.title('Image (it = %i)'%(i+1))

            plt.subplot(224)
            plt.imshow(np.log10(np.abs(fft_im_sr)), 'gray')
            plt.title('FFT image (log10) (it = %i)'%(i+1))

            plt.show()
            plt.close()

            # filter
            plt.figure()
            plt.imshow(filter, 'gray')
            plt.title('Filter (sigma = %0.3f)'%(sigma))
            plt.show()
            plt.close()
            exit()

        # save_path = os.path.join(save_dir, 'it_'+str(i+1))
        # if not(os.path.exists(save_path)):
        #     os.makedirs(save_path)

        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(old_im_sr.real, 'gray')
        # plt.title('Image (it = %i)'%(i+1))
        # plt.colorbar()

        # plt.subplot(222)
        # plt.imshow(im_sr.real, 'gray')
        # plt.title('SR before forced (it = %i)'%(i+1))
        # plt.colorbar()

        # save_im(os.path.join(save_path,'sr_image_old.png'), old_im_sr.real)
        # save_im(os.path.join(save_path,'sr_image_before_forced.png'), im_sr.real)


        im_sr[HR_grid>0] = HR_grid[HR_grid>0]
        if plot_debug_idx!=None:
            intensity[i+1] = im_sr[plot_debug_idx[0], plot_debug_idx[0]].real
        # plt.subplot(223)
        # plt.imshow(im_sr.real, 'gray')
        # plt.title('SR (it = %i)'%(i+1))
        # plt.colorbar()

        # plt.show()
        # plt.savefig(os.path.join(save_path, 'plot.png'), dpi=200)
        # plt.close()
        # save_im(os.path.join(save_path,'sr_image.png'), im_sr.real)
    

        if MSE!=None:
            err[i] = np.sum((im_sr.real - old_im_sr.real)**2)/(im_sr.shape[0]*im_sr.shape[1])
        # max_val = np.amax(abs(im_sr))
        # min_val = np.amin(abs(im_sr))
        # im_sr = 255*(abs(im_sr) - min_val) / (max_val-min_val)

        
        if intermediary_step and (i+1)%int(it/10) == 0:
            save_path = os.path.join(save_dir, 'it_'+str(i+1))
            if not(os.path.exists(save_path)):
                os.makedirs(save_path)
            save_im(os.path.join(save_path,'sr_image_new.png'), im_sr.real)
    
    if MSE!=None:
        plt.figure(figsize=(10,7))
        plt.plot(np.linspace(1,it,it), err)
        plt.yscale('log')
        plt.grid(True, axis='both', which='both')
        plt.title('Mean Square Error')
        plt.xlabel('iterations')
        plt.ylabel('MSE')
        if save_dir == None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, 'MSE.png'), dpi=200)
        plt.close()
    if plot_debug_idx!=None:
        plt.figure(figsize=(10,7))
        plt.plot(np.linspace(1,it+1,it+1), intensity)
        plt.grid(True, axis='both', which='both')
        plt.title('Intensity of pixel (%i,%i)'%(plot_debug_idx[0], plot_debug_idx[1]))
        plt.xlabel('iterations')
        plt.ylabel('Intensity')
        if save_dir == None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, 'intensity(%ix%i).png'%(plot_debug_idx[0], plot_debug_idx[1])), dpi=200)
        plt.close()
    
        
    # global_time = time.time() - global_start_time
    # print('Execution time : %0.2f' % (global_time))
    if not(out_filter) or not(psf):
        return im_sr
    else:
        return im_sr, filter
