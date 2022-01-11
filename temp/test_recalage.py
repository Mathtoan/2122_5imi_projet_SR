import cv2
import numpy as np
import matplotlib.pyplot as plt



def shifts_calculation(im_ref, im_recal, display=False):
    h,w = im_ref.shape
    orb_detector = cv2.ORB_create(5000)
    kp_recal, d_recal = orb_detector.detectAndCompute(im_recal, None)
    kp_ref, d_ref = orb_detector.detectAndCompute(im_ref, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(d_recal, d_ref)
    matches = sorted(matches,key = lambda x: x.distance)
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp_recal[matches[i].queryIdx].pt
        p2[i, :] = kp_ref[matches[i].trainIdx].pt
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    shift_x = np.zeros([h,w])
    shift_y = np.zeros([h,w])
    recal = np.zeros([h,w])

    homography = np.linalg.inv(homography)
    for i in range(h):
        for j in range(w):
            [x_prime, y_prime, _] = np.matmul(homography,[i,j,1])/np.matmul(homography[2],[i,j,1])
            vec_to_normalize = [x_prime, y_prime, 1]
            n = np.linalg.norm(vec_to_normalize)
            if n != 0: 
                vec_to_normalize =  vec_to_normalize / n
                x_prime = vec_to_normalize[0]
                y_prime = vec_to_normalize[1]
            shift_x[i][j] = x_prime - i
            shift_y[i][j] = y_prime - j


    
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



##Recalage image 1
# Open the image files.
img1_color = cv2.imread("temp/im_test_1.jpg") # Image to be aligned.
img2_color = cv2.imread("temp/im_ref.jpg") # Reference image.

# plt.figure()
# plt.imshow(img1_color)
# plt.show()

# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

sx,sy = shifts_calculation(img2,img1)
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

exit(1)




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


