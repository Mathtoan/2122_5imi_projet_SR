from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'src')
from utils import *
import scipy as sp
from skimage import transform




def shifts_calculation(im_ref, im_recal, display=False):
    h,w = im_ref.shape

    #Recalage 1
    im_recal_cv = img_as_ubyte(im_recal)
    im_ref_cv = img_as_ubyte(im_ref)
    nb_POI = 5000
    orb_detector = cv2.ORB_create(nb_POI)
    contour_recal = cv2.Laplacian(im_recal_cv,cv2.CV_8UC1)
    contour_ref = cv2.Laplacian(im_ref_cv,cv2.CV_8UC1)
    plt.figure()
    plt.subplot(121)
    plt.imshow(contour_recal,'gray')
    plt.subplot(122)
    plt.imshow(contour_ref,'gray')
    plt.show()
    kp_recal, d_recal = orb_detector.detectAndCompute(contour_recal, None)
    kp_ref, d_ref = orb_detector.detectAndCompute(contour_ref, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(d_recal, d_ref)
    matches = sorted(matches,key = lambda x: x.distance)
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp_recal[matches[i].queryIdx].pt
        p2[i, :] = kp_ref[matches[i].trainIdx].pt
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    transformed_img = cv2.warpPerspective(im_recal, homography, (w, h))
    plt.imsave('temp/im_warpPerspective1.jpg', transformed_img, cmap='gray')
    diff = abs(im_ref-transformed_img)
    print("Différence recalage pour nb_POI=" + str(nb_POI) + " : ", np.mean(diff))
    plt.figure()
    plt.imshow(diff,'gray')
    plt.colorbar()
    plt.savefig("temp/diff_ref_recalee1.jpg")
    # plt.show()

    # #Recalages suivants :
    # for k in range(10):
    #     addx = 10
    #     addy = 10
    #     x = randint(0,addx)
    #     y = randint(0,addy)
    #     newGrid = np.zeros([h+addy,w+addx])
    #     for l in range(h):
    #         for m in range(w):
    #             newGrid[l+addy,m+addx] = transformed_img[l,m]
    #     transformed_img_cv = img_as_ubyte(newGrid)
    #     orb_detector = cv2.ORB_create(5000)
    #     kp_recal, d_recal = orb_detector.detectAndCompute(transformed_img_cv, None)
    #     kp_ref, d_ref = orb_detector.detectAndCompute(im_ref_cv, None)
    #     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    #     matches = matcher.match(d_recal, d_ref)
    #     matches = sorted(matches,key = lambda x: x.distance)
    #     matches = matches[:int(len(matches)*0.9)]
    #     no_of_matches = len(matches)
    #     p1 = np.zeros((no_of_matches, 2))
    #     p2 = np.zeros((no_of_matches, 2))
    #     for i in range(len(matches)):
    #         p1[i, :] = kp_recal[matches[i].queryIdx].pt
    #         p2[i, :] = kp_ref[matches[i].trainIdx].pt
    #     homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    #     transformed_img = cv2.warpPerspective(newGrid, homography, (w, h))
    #     str_name1 = "temp/im_warpPerspective" + str(k+2) + ".jpg"
    #     plt.imsave(str_name1, transformed_img, cmap='gray')
    #     diff = abs(im_ref-transformed_img)
    #     print("Différence recalage n°", str(k+2)," : ", np.mean(diff))
    #     plt.figure()
    #     plt.imshow(diff,'gray')
    #     plt.colorbar()
    #     str_name2 = "temp/diff_ref_recalee" + str(k+2) + ".jpg"
    #     plt.savefig(str_name2)




    shift_x = np.zeros([h,w])
    shift_y = np.zeros([h,w])
    recal   = np.zeros([h,w])

    homography = np.linalg.inv(homography)
    print(homography)
    for i in range(h):
        for j in range(w):
            [x_prime, y_prime, s] = np.matmul(homography,[i,j,1])/np.matmul(homography[2],[i,j,1])
            x_prime = int(np.floor(x_prime/s))
            y_prime = int(np.floor(y_prime/s))
            
            if x_prime>0 and x_prime<h and y_prime>0 and y_prime<w:
                recal[x_prime,y_prime] = im_recal[i,j]

            shift_x[i][j] = x_prime - i
            shift_y[i][j] = y_prime - j
    plt.imsave('temp/im_homography_1.png', recal, cmap='gray')


    
    if(display):
        plt.figure()
        plt.subplot(121)
        plt.imshow(shift_x)
        plt.title("Shifts selon x")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(shift_y)
        plt.title("Shifts selon y")
        plt.colorbar()
        plt.show()

    return shift_x,shift_y



def Create_HR_GRID(list_im, upscale_factor):
    print('Registration using handmade method')
    #Variables et images
    im_ref = list_im[0]
    height, width = im_ref.shape
    im_ref_cv = img_as_ubyte(im_ref)

    #Grille HR vide
    new_height = height*upscale_factor
    new_width  = width*upscale_factor
    HR_Grid = np.zeros([new_height,new_width])
    for h in range(height):
        for w in range(width):
            idx_h_ref = h*upscale_factor
            idx_w_ref = w*upscale_factor
            HR_Grid[idx_h_ref][idx_w_ref] = im_ref[h][w]

    for i in range(1,len(list_im)):
        current_im = list_im[i]
        im_to_register_cv = img_as_ubyte(current_im)
        orb_detector = cv2.ORB_create(5000)
        kp_recal, d_recal = orb_detector.detectAndCompute(im_to_register_cv, None)
        kp_ref, d_ref = orb_detector.detectAndCompute(im_ref_cv, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = matcher.match(d_recal, d_ref)
        matches = sorted(matches,key = lambda x: x.distance)
        no_of_matches = len(matches)
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
        for k in range(len(matches)):
            p1[k, :] = kp_recal[matches[k].queryIdx].pt
            p2[k, :] = kp_ref[matches[k].trainIdx].pt

        #Test skimage.transform#
        transfo = transform.estimate_transform('affine', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_affine_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_affine_"+str(i)+".png",diff,cmap='gray')

        transfo = transform.estimate_transform('similarity', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_similarity_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_similarity_"+str(i)+".png",diff,cmap='gray')

        transfo = transform.estimate_transform('piecewise-affine', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_piecewise-affine_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_piecewise-affine_"+str(i)+".png",diff,cmap='gray')

        transfo = transform.estimate_transform('projective', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_projective_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_projective_"+str(i)+".png",diff,cmap='gray')

        transfo = transform.estimate_transform('polynomial', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_polynomial_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_polynomial_"+str(i)+".png",diff,cmap='gray')

        transfo = transform.estimate_transform('euclidean', p1, p2)
        transformed = transform.warp(current_im, transfo)
        diff = abs(current_im - transformed)
        plt.imsave("temp/recalage_euclidean_"+str(i)+".png",transformed,cmap='gray')
        plt.imsave("temp/diff_euclidean_"+str(i)+".png",diff,cmap='gray')
        #########################

        homography, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
        homography[0,2] = homography[0,2]*upscale_factor
        homography[1,2] = homography[1,2]*upscale_factor
        homography = np.linalg.inv(homography)
        for i in range(height):
            for j in range(width):
                [x_prime, y_prime, _] = np.matmul(homography,[i,j,1])/np.matmul(homography[2],[i,j,1])
                
                x_prime_decimal, x_prime_int = math.modf(x_prime)
                y_prime_decimal, y_prime_int = math.modf(y_prime)
                x_prime_int = int(x_prime_int)*upscale_factor
                y_prime_int = int(y_prime_int)*upscale_factor

                # POINT CENTRAL
                x_point_central = x_prime_int
                y_point_central = y_prime_int
                alpha = (1-x_prime_decimal) * (1-y_prime_decimal)
                if x_point_central>0 and x_point_central<new_height and y_point_central>0 and y_point_central<new_width:# and HR_Grid[x_point_central,y_point_central] ==0:
                    HR_Grid[x_point_central,y_point_central] = alpha * current_im[i,j]
                # POINT VERTICAL
                x_point_vertical = x_prime_int + 1 
                y_point_vertical = y_prime_int
                alpha = (x_prime_decimal) * (1-y_prime_decimal)
                if x_point_vertical>0 and x_point_vertical<new_height and y_point_vertical>0 and y_point_vertical<new_width:# and HR_Grid[x_point_vertical,y_point_vertical] ==0:
                    HR_Grid[x_point_vertical,y_point_vertical] = alpha * current_im[i,j]
                # POINT HORIZONTAL
                x_point_horizontal = x_prime_int
                y_point_horizontal = y_prime_int + 1 
                alpha = (1-x_prime_decimal) * (y_prime_decimal)
                if x_point_horizontal>0 and x_point_horizontal<new_height and y_point_horizontal>0 and y_point_horizontal<new_width:# and HR_Grid[x_point_horizontal,y_point_horizontal] ==0:
                    HR_Grid[x_point_horizontal,y_point_horizontal] = alpha * current_im[i,j]           
                # POINT DIAGONAL
                x_point_diagonal = x_prime_int + 1
                y_point_diagonal = y_prime_int + 1
                alpha = (x_prime_decimal) * (y_prime_decimal)
                if x_point_diagonal>0 and x_point_diagonal<new_height and y_point_diagonal>0 and y_point_diagonal<new_width:# and HR_Grid[x_point_diagonal,y_point_diagonal] ==0:
                    HR_Grid[x_point_diagonal,y_point_diagonal] = alpha * current_im[i,j]  

                # print("i,j:",i,",",j," - x:",x_point_central,",",x_point_vertical,",",x_point_horizontal,",",x_point_diagonal,
                #                      " - y:",y_point_central,",",y_point_vertical,",",y_point_horizontal,",",y_point_diagonal)

    plt.imsave('temp/HR_Grid.png', HR_Grid, cmap='gray')
    return HR_Grid


#Pixel based registration
# img1_color = io.imread("temp/im_test_1.jpg")
# img2_color = io.imread("temp/im_ref.jpg")
img1_color = io.imread("fig/iPhone13Pro/scene1/IMG_1330.JPG")
img2_color = io.imread("fig/iPhone13Pro/scene1/IMG_1331.JPG")
img3_color = io.imread("fig/iPhone13Pro/scene1/IMG_1332.JPG")
img1 = rgb2gray(img1_color)
img2 = rgb2gray(img2_color)
img3 = rgb2gray(img3_color)
scale_fac = 4
img1 = rescale(img1,1/scale_fac)
img2 = rescale(img2,1/scale_fac)
img3 = rescale(img3,1/scale_fac)
l = [img1, img2, img3]
grid = Create_HR_GRID(l,scale_fac)
exit()


diff = abs(img2 - img1)
plt.figure()
plt.imshow(diff,'gray')
plt.colorbar()
plt.show()


exit()
sx,sy = shifts_calculation(img2,img1)
exit()

scale_fac = 1
im_test = rescale(img1, 1/scale_fac)
im_ref = rescale(img2, 1/scale_fac)
# im_test = img1
# im_ref = img2
calc_shifts, error, phasediff = phase_cross_correlation(im_test,im_ref,upsample_factor=scale_fac)
print(calc_shifts, " ", error, " ", phasediff)
registered_im = rotate(shift(im_test, shift=(calc_shifts[0], calc_shifts[1]), mode='constant'),-calc_shifts[0])
plt.figure()
plt.subplot(131)
plt.imshow(im_ref,'gray')
plt.title("im ref")
plt.subplot(132)
plt.imshow(im_test,'gray')
plt.title("im test")
plt.subplot(133)
plt.imshow(registered_im,'gray')
plt.title("registered im")
plt.show()

exit()
result_1 = sp.signal.correlate2d(im_test,im_ref,boundary='symm', mode='same')
result_2 = sp.signal.correlate2d(im_test,im_ref,boundary='symm', mode='full')
# result_3 = sp.signal.correlate2d(im_test,im_ref,boundary='symm', mode='valid')
result_4 = sp.signal.correlate2d(im_test,im_ref,boundary='wrap', mode='same')
result_5 = sp.signal.correlate2d(im_test,im_ref,boundary='wrap', mode='full')
# result_6 = sp.signal.correlate2d(im_test,im_ref,boundary='wrap', mode='valid')
result_7 = sp.signal.correlate2d(im_test,im_ref,boundary='fill', mode='same')
result_8 = sp.signal.correlate2d(im_test,im_ref,boundary='fill', mode='full')
# result_9 = sp.signal.correlate2d(im_test,im_ref,boundary='fill', mode='valid')
plt.figure()
plt.subplot(321)
plt.imshow(result_1,'gray')
plt.subplot(322)
plt.imshow(result_2,'gray')
plt.subplot(323)
plt.imshow(result_4,'gray')
plt.subplot(324)
plt.imshow(result_5,'gray')
plt.subplot(325)
plt.imshow(result_7,'gray')
plt.subplot(326)
plt.imshow(result_8,'gray')
plt.show()
exit()

h,w,_ = img2_color.shape
img2_color = img2_color[100:w-100,100:h-100]
plt.figure()
plt.subplot(121)
plt.imshow(img1_color)
plt.subplot(122)
plt.imshow(img2_color)
plt.show()
result = cv2.matchTemplate(img1_color,img2_color,cv2.TM_SQDIFF_NORMED)
plt.figure()
plt.imshow(result)
plt.show()
exit()







# plt.figure()
# plt.imshow(img1_color)
# plt.show()

img1_color = cv2.imread("temp/im_test_1.jpg")
img2_color = cv2.imread("temp/im_ref.jpg")


# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

sx,sy = shifts_calculation(img2,img1)
exit(1)

h,w = img2.shape
im_recalee = np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if i-sx[i,j]>0 and i-sx[i,j]<h and j-sy[i,j]>0 and j-sy[i,j]<w: 
            im_recalee[i,j] = img1[int(i-sx[i,j]),int(j-sy[i,j])]

plt.figure()
plt.imshow(im_recalee, 'gray')
plt.imsave('temp/im_recalee.png', im_recalee, cmap='gray')
# plt.show()
# plt.close()






# img1 = img1_color[0]
# img2 = img2_color[0]
height, width = img2.shape

# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)

# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
# (which is not required in this case).
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match the two sets of descriptors.
matches = matcher.match(d1, d2)

# Sort matches on the basis of their Hamming distance.
matches = sorted(matches,key = lambda x: x.distance)

# Take the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
print(homography)

for i in range(height):
    for j in range(width):
        x_prime = homography[0][0]*i + homography[0][1]*j + homography[0][2]
        y_prime = homography[1][0]*i + homography[1][1]*j + homography[1][2]
        shift_x = x_prime - i
        shift_y = y_prime - j
    # print("shift x = " + str(shift_x) + " et shift y = ", str(shift_y))

# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,
					homography, (width, height))

# Save the output.
plt.imsave('temp/im_1_recalee.jpg', transformed_img, cmap='gray')





