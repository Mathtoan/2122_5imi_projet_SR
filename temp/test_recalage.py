import cv2
import numpy as np
import matplotlib.pyplot as plt

##Recalage image 1
# Open the image files.
img1_color = cv2.imread("im_test_1.jpg") # Image to be aligned.
img2_color = cv2.imread("im_ref.jpg") # Reference image.

# plt.figure()
# plt.imshow(img1_color)
# plt.show()

# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
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
    print("shift x = " + str(shift_x) + " et shift y = ", str(shift_y))

# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,
					homography, (width, height))

# Save the output.
cv2.imwrite('im_1_recalee.jpg', transformed_img)


# ##Recalage image 2
# # Open the image files.
# img1_color = cv2.imread("im_test_2.jpg") # Image to be aligned.
# img2_color = cv2.imread("im_ref.jpg") # Reference image.

# # Convert to grayscale.
# img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
# height, width = img2.shape

# # Create ORB detector with 5000 features.
# orb_detector = cv2.ORB_create(5000)

# # Find keypoints and descriptors.
# # The first arg is the image, second arg is the mask
# # (which is not required in this case).
# kp1, d1 = orb_detector.detectAndCompute(img1, None)
# kp2, d2 = orb_detector.detectAndCompute(img2, None)

# # Match features between the two images.
# # We create a Brute Force matcher with
# # Hamming distance as measurement mode.
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
# # Match the two sets of descriptors.
# matches = matcher.match(d1, d2)

# # Sort matches on the basis of their Hamming distance.
# matches = sorted(matches,key = lambda x: x.distance)

# # Take the top 90 % matches forward.
# matches = matches[:int(len(matches)*0.9)]
# no_of_matches = len(matches)

# # Define empty matrices of shape no_of_matches * 2.
# p1 = np.zeros((no_of_matches, 2))
# p2 = np.zeros((no_of_matches, 2))

# for i in range(len(matches)):
#     p1[i, :] = kp1[matches[i].queryIdx].pt
#     p2[i, :] = kp2[matches[i].trainIdx].pt

# # Find the homography matrix.
# homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

# # Use this matrix to transform the
# # colored image wrt the reference image.
# transformed_img = cv2.warpPerspective(img1_color,
# 					homography, (width, height))

# # Save the output.
# cv2.imwrite('im_2_recalee.jpg', transformed_img)

