import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import itk
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from random import randint
from scipy.ndimage import shift
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

def format_time(t):
    ms = int((t - int(t))*100)
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '{0:d}:{1:02d}:{2:02d}.{3:03d}'.format(h, m, s, ms)
    else:
        return '{0:02d}:{1:02d}.{2:03d}'.format(m, s, ms)

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

def centered_square(h, w,sigma):
    u = np.linspace(-(w-1)/2.,(w-1)/2., w)/((w-1)/2)
    v = np.linspace(-(h-1)/2.,(h-1)/2., h)/((h-1)/2)
    U,V = np.meshgrid(u,v)

    H = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if abs(U[i][j])<sigma and abs(V[i][j])<sigma:
                H[i][j] = 1.
    return H

def centered_circle(h,w,sigma):
    # 'Normalized [-1,1] meshgrids' 
    u = np.linspace(-(w-1)/2.,(w-1)/2., w)/((w-1)/2)
    v = np.linspace(-(h-1)/2.,(h-1)/2., h)/((h-1)/2)
    U,V = np.meshgrid(u,v)

    H = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if U[i][j]**2 + V[i][j]**2 < sigma**2:
                H[i][j] = 1
    return H

def image_histogram(im, title, bins=np.linspace(0,1,256), save_dir=None):
    plt.figure(figsize=(10,7))
    plt.hist(im.reshape((im.shape[0]*im.shape[1],)), bins=bins)
    plt.title(title)
    plt.grid(True, axis='y')
    if save_dir==None:
        plt.show()
    else:
        plt.savefig(save_dir, dpi=200)
    plt.close()
    
def creation_HR_grid(im_ref, list_image_input_dir, idx_ref, upscale_factor, method, color):
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
    im_ref_up = np.zeros(sr_size)
    cpt_grid = np.zeros(sr_size)
    print("sr size = ", sr_size)

    for h in range(lr_size[0]):
        for w in range(lr_size[1]):
    
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor

            im_ref_up[idx_h_ref][idx_w_ref] = im_ref[h][w]
    im_sr = np.copy(im_ref_up)
    cpt_grid[im_ref_up != 0] += 1
    image_histogram(im_sr, 'im_sr_0', bins=np.linspace(1/255,1,255), save_dir='output/hist/im_sr_0')

    # print(np.amin(im_sr), np.amax(im_sr))
    print('######### idx_ref =', idx_ref, '#########')
    for k in tqdm(range(len(list_image_input_dir)), desc='Registration'):
        if k != idx_ref:
            im_to_register = io.imread(list_image_input_dir[k])
            if color=='gray':
                im_to_register = rgb2gray(im_to_register)
            im_to_register = rescale(im_to_register, 1/upscale_factor)
            registered_im = eval('computing_regitration_'+method+'(im_ref, im_to_register, upscale_factor)')

            print("ref: ",im_ref.shape)
            print("registered: ",registered_im.shape)
            # plt.imsave("registered_"+str(k)+".png",im_ref+registered_im, cmap='gray')
            
            # im_ref_up[:10,:10] = 1
            # registered_im[sr_size[0]-10:, sr_size[1]-10:] = 1
            plt.imsave("registered_"+str(k)+".png",im_ref_up+registered_im, cmap='gray')
            
            # im_sr[im_sr==0] = registered_im[im_sr==0]
            im_sr += registered_im
            cpt_grid[registered_im != 0] += 1
            
            save_sr = np.copy(im_sr)
            save_sr[cpt_grid!=0] = save_sr[cpt_grid!=0]/cpt_grid[cpt_grid!=0]
            save_sr = save_sr.reshape(sr_size)
            print(save_sr.shape)
            
            image_histogram(save_sr, 'im_sr_'+str(k), bins=np.linspace(1/255,1,255), save_dir='output/hist/im_sr_'+str(k))
            
            # exit()
    
    im_sr[cpt_grid!=0] = im_sr[cpt_grid!=0] / cpt_grid[cpt_grid!=0]
            
    global_time_str = format_time(time.time() - global_start_time)
    print('Execution time : ' + global_time_str)
    return im_sr

def computing_regitration_translation(im_ref, im_to_register, upscale_factor, display=False):
    print('Registration using translation method')
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
    print('Registration using POI method')
    
    height, width = im_ref.shape
    im_to_register_cv = img_as_ubyte(im_to_register)
    im_ref_cv = img_as_ubyte(im_ref)
    
    offset = np.eye(3)
    offset_x = height*upscale_factor/2
    offset_y = width*upscale_factor/2
    offset[0,2] = offset_x
    offset[1,2] = offset_y
    
    T = np.float32([[1, 0, -height/2], [0, 1, -width/2]])
    im_to_register_cv = cv2.warpAffine(im_to_register_cv, T, (width, height))
    im_ref_cv = cv2.warpAffine(im_ref_cv, T, (width, height))

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
    homography[:2,2] = homography[:2,2]*upscale_factor
    
    registered_im = np.zeros([height*upscale_factor,width*upscale_factor])
    for h in range(height):
        for w in range(width):
            [x_prime, y_prime, s] = np.matmul(homography,[h*upscale_factor,w*upscale_factor,1])/np.matmul(homography[2],[h*upscale_factor,w*upscale_factor,1])
            idx_h_ref = int(x_prime)
            idx_w_ref = int(y_prime)
            if idx_h_ref>0 and idx_h_ref<height*upscale_factor and idx_w_ref>0 and idx_w_ref<width*upscale_factor:
                registered_im[idx_h_ref][idx_w_ref] = im_to_register[h][w]
    
    
    # im_to_register_up = np.zeros([height*upscale_factor,width*upscale_factor])
    # for h in range(height):
    #     for w in range(width):
    #         idx_h_ref = h*upscale_factor
    #         idx_w_ref = w*upscale_factor
    #         im_to_register_up[idx_h_ref][idx_w_ref] = im_to_register[h][w]     
    # registered_im = cv2.warpPerspective(im_to_register_up, homography, (width*upscale_factor, height*upscale_factor), cv2.INTER_LINEAR)

    
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
    
    # plt.imsave('registered.png', registered_im, cmap='gray')
    # exit()

    return registered_im

def computing_regitration_pixel(im_ref, im_to_register, upscale_factor, display=False):
    print('Registration using intensity pixel method')
    
    height, width = im_ref.shape
    # im_to_register_cv = img_as_ubyte(im_to_register)
    # im_ref_cv = img_as_ubyte(im_ref)
    
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

def computing_regitration_itk(im_ref, im_to_register, upscale_factor, display=False):
    print('Registration using itk method')
    
    height, width = im_ref.shape

    # ----------------------
    # Lecture des images
    # ----------------------
    
    im_ref_itk = itk.GetImageFromArray(im_ref)
    im_to_register_itk = itk.GetImageFromArray(im_to_register)
    im_ref_itk_type = type(im_ref_itk) # On récupère le type de l'image fixe
    im_to_register_itk_type = type(im_to_register_itk) # On récupère le type de l'image translatée
    
    im_to_register_up = np.zeros([height*upscale_factor,width*upscale_factor])
    for h in range(height):
        for w in range(width):
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor
            im_to_register_up[idx_h_ref][idx_w_ref] = im_to_register[h][w]
    im_to_register_up_itk = itk.GetImageFromArray(im_to_register_up)

    # ----------------------
    # Optimiseur
    # ----------------------
    optimizer = itk.RegularStepGradientDescentOptimizer.New() # Instance de la classe d'optimiseur choisie
    optimizer.SetMaximumStepLength(.1)
    optimizer.SetMinimumStepLength(.001)
    optimizer.SetNumberOfIterations(1000)
    # optimizer.SetNumberOfIterations(200)
    optimizer.SetScales([500, 1, 1, 1, 1])
    # ----------------------
    # initial Parameter
    # ----------------------

    initialTransform = itk.CenteredRigid2DTransform[itk.D].New() # Instance de la classe de transformation choisie
    initialParameters = initialTransform.GetParameters() # Récupération des paramètres de la transformation

    initialParameters[0] = 0
    initialParameters[1] = itk.size(im_to_register_itk)[0]/2
    initialParameters[2] = itk.size(im_to_register_itk)[1]/2
    initialParameters[3] = 0
    initialParameters[4] = 0

    # ----------------------
    # Interpolateur
    # ----------------------

    interpolator = itk.NearestNeighborInterpolateImageFunction[im_to_register_itk_type, itk.D].New()

    # ----------------------
    # Metrics
    # ----------------------
    metric = itk.MeanSquaresImageToImageMetric[im_ref_itk_type, im_to_register_itk_type].New()

    # ----------------------
    # Exécution du recalage
    # ----------------------

    registration_filter = itk.ImageRegistrationMethod[im_ref_itk_type, im_to_register_itk_type].New() # Instance de la classe de recalage
    registration_filter.SetFixedImage(im_ref_itk) # Image de référence
    registration_filter.SetMovingImage(im_to_register_itk) # Image à recaler
    registration_filter.SetOptimizer(optimizer) # Optimiseur
    registration_filter.SetTransform(itk.CenteredRigid2DTransform[itk.D].New()) # Transformation
    registration_filter.SetInitialTransformParameters(initialParameters) #Application de la transformation initiale
    registration_filter.SetInterpolator(interpolator) # Interpolateur
    registration_filter.SetMetric(metric) # Métrique
    registration_filter.Update() # Exécution du recalage

    # ----------------------
    # final Parameter upscaled
    # ----------------------
    
    final_transform = registration_filter.GetTransform()
    finalParameters = final_transform.GetParameters() # Récupération des paramètres de la transformation

    finalParameters[1] = itk.size(im_to_register_up_itk)[0]/2
    finalParameters[2] = itk.size(im_to_register_up_itk)[1]/2
    finalParameters[3] = finalParameters[3]*upscale_factor
    finalParameters[4] = finalParameters[4]*upscale_factor
    
    final_transform.SetParameters(finalParameters)
    
    # ----------------------
    # Apply last transform
    # ----------------------
    
    test_im = np.zeros(im_to_register_itk.GetLargestPossibleRegion().GetSize())
    test_im[50:60,50:60] = 1
    test_im = itk.GetImageFromArray(test_im)
    test_im_type = type(test_im)

    # resample_filter = itk.ResampleImageFilter[im_ref_itk_type,test_im_type].New() #Instance de la classe de ré-échantillonnage
    # resample_filter.SetInput(test_im) # Image d'entrée
    # resample_filter.SetTransform(final_transform)
    # resample_filter.SetSize(test_im.GetLargestPossibleRegion().GetSize())
    
    resample_filter = itk.ResampleImageFilter[im_ref_itk_type,im_to_register_itk_type].New() #Instance de la classe de ré-échantillonnage
    resample_filter.SetInput(im_to_register_up_itk) # Image d'entrée
    resample_filter.SetTransform(final_transform)
    resample_filter.SetSize(im_to_register_up_itk.GetLargestPossibleRegion().GetSize())

    registered_im_itk = resample_filter.GetOutput()
    registered_im = itk.GetArrayFromImage(registered_im_itk)
    
    
    for i in range(registered_im.shape[0]-1):
        for j in range(registered_im.shape[1]-1):
            if registered_im[i,j] != 0 and (registered_im[i+1,j] != 0 or registered_im[i,j+1] != 0):
                int1 = registered_im[i,j]
                int2 = registered_im[i+1,j]
                int3 = registered_im[i,j+1]
                int4 = registered_im[i+1,j+1]
                liste_int = [int1,int2,int3,int4]
                liste_pts = [[i,j],[i+1,j],[i,j+1],[i+1,j+1]]
                int_max = max(liste_int)
                ind = liste_int.index(int_max)
                int_tot = np.sum(liste_int)
                x = liste_pts[ind][0]
                y = liste_pts[ind][1]
                for k in range(4):
                    x2 = liste_pts[k][0]
                    y2 = liste_pts[k][1]
                    registered_im[x2,y2] = 0.0
                registered_im[x,y] = int_tot
                
                
    # plt.imsave('test.png',registered_im,cmap='gray')
    # exit()
    
    print(optimizer.GetCurrentIteration())
    print(optimizer.GetValue())
    
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


def PG_method(HR_grid, sigma,
              out_filter=False, intermediary_step=None, eps=1e-5,save_dir=None, psf=False,
              imshow_debug=False, plot_debug_intensity=False, max_steps=100, filter_type='centered_circle'):
    
    if intermediary_step!=None:
        if save_dir==None:
            print("A save path should be input : won't save intermediary steps")
            intermediary_step = None
    
    global_start_time = time.time()
    print('---- Papoulis–Gerchberg method ----')
    print("initialization : generating filter")
    initialization_start_time = time.time()

    im_sr = HR_grid
    sr_size = HR_grid.shape
    print(HR_grid.shape)

    if filter_type == 'centered_circle' or filter_type == 'circle':
        filter = centered_circle(sr_size[0], sr_size[1],sigma)
    elif filter_type == 'centered_square' or filter_type == 'square':
        filter = centered_square(sr_size[0], sr_size[1],sigma)
    elif filter_type == 'gaussian' or filter_type == 'gaussian_2d':
        filter = gaussian_2d(sr_size[0], sr_size[1], sigma, sigma)
    else:
        raise ValueError("Unknown filter")
    
    initialization_time_str = format_time(time.time() - initialization_start_time)
    global_time_str = format_time(time.time() - global_start_time)
    print('Execution time : ' + initialization_time_str + ' | Total execution time : ' + global_time_str)

    err = eps+1e3
    MSE = []
    if plot_debug_intensity:
        u,v = np.where(HR_grid==0)
        r = randint(0,len(u)-1)

        plot_debug_idx = (u[r], v[r])
        intensity = [0]

    i = 0
    while (err>eps or i<2) and i<max_steps:
        i+=1
            
        interation_start_time = time.time()
        print('step = %i'%(i))
        old_im_sr = im_sr.real

        # -------------
        # Apply filter
        # -------------

        # In Fourier domain
        if not(psf): 
            old_fft_im_sr = fftshift(fft2(old_im_sr))

            if i==0 and save_dir!=None:
                save_im(os.path.join(save_dir, 'fft.png'), np.log10(np.abs(old_fft_im_sr)))
                
            fft_im_sr = np.multiply(old_fft_im_sr, filter)

            im_sr = ifft2(ifftshift(fft_im_sr))

        # In direct space
        else: 
            im_sr = gaussian(old_im_sr, sigma)

        # ---------------------------
        # Plotting for debug purposes
        # ---------------------------
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


        # -----------------------------
        # Set known values from HR_grid
        # -----------------------------
        im_sr[HR_grid>0] = HR_grid[HR_grid>0]
        if plot_debug_intensity:
            intensity.append(im_sr[plot_debug_idx].real)

        # -------------------------------
        # Computing MSE and stop criteria
        # -------------------------------
        # MSE.append(np.sum((im_sr.real - old_im_sr.real)**2)/(im_sr.shape[0]*im_sr.shape[1]))
        MSE.append(np.amax((im_sr.real - old_im_sr.real)**2))

        if len(MSE)>1:
            err = abs(MSE[-1]-MSE[-2])

        # -------------------------
        # Saving intermediary steps
        # -------------------------
        if intermediary_step!=None:
            if (i)%intermediary_step == 0:
                print("Saving step : %i"%(i))
                save_path = os.path.join(save_dir, 'it_'+str(i))
                if not(os.path.exists(save_path)):
                    os.makedirs(save_path)
                save_im(os.path.join(save_path,'sr_image_new.png'), im_sr.real)
                image_histogram(im_sr, 'Histogram SR Image (it=%i)'%(i), save_dir=os.path.join(save_path,'hist_im_sr.png'))

        interation_time_str = format_time(time.time() - interation_start_time)
        global_time_str = format_time(time.time() - global_start_time)


        print('Execution time : ' + interation_time_str + ' | Total execution time : ' + global_time_str)
    
    if i==max_steps:
        print("/!\ Max steps reached /!\ ")

    save_path = os.path.join(save_dir, 'it_'+str(i))
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    
    # --------------
    # Saving results
    # --------------
    save_im(os.path.join(save_path,'sr_image_old.png'), old_im_sr.real)
    save_im(os.path.join(save_path, 'fft_before_filter.png'), np.log10(np.abs(old_fft_im_sr)))
    save_im(os.path.join(save_path, 'fft_after_filter.png'), np.log10(np.abs(fft_im_sr)))
    save_im(os.path.join(save_path,'sr_image.png'), im_sr.real)

    
    # ------------
    # Plotting MSE
    # ------------
    plt.figure(figsize=(10,7))
    plt.plot(np.linspace(1,i,i), MSE)
    plt.grid(True, axis='both', which='both')
    plt.title('Max of Squared Errors')
    plt.xlabel('iterations')
    if save_dir == None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, 'MSE.png'), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Plotting evolution of a random pixel set a 0 in initilization
    # -------------------------------------------------------------
    if plot_debug_intensity:
        plt.figure(figsize=(10,7))
        plt.plot(np.linspace(0,i,i+1), intensity)
        plt.axis([0, i, 0, 1])
        plt.grid(True, axis='both', which='both')
        plt.title('Intensity of pixel (%i,%i)'%(plot_debug_idx[0], plot_debug_idx[1]))
        plt.xlabel('iterations')
        plt.ylabel('Intensity')
        if save_dir == None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, 'intensity(%ix%i).png'%(plot_debug_idx[0], plot_debug_idx[1])), dpi=200)
        plt.close()
    
    if not(out_filter) or psf:
        return im_sr
    else:
        return im_sr, filter
