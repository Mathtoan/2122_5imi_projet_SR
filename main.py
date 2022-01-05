import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from skimage import io
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation

# Path
device = 'iPhone13Pro'
scene = 'scene1'
input_dir = os.path.join('fig', device, scene)


list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()

upscale_factor = 4

# Registration and SR grid creation
for idx_ref in range(len(list_image_input_dir)):
    print('######### idx_ref =', idx_ref, '#########')
    im_ref = rgb2gray(io.imread(list_image_input_dir[idx_ref]))
    lr_size = im_ref.shape
    sr_size = [lr_size[0]*upscale_factor, lr_size[1]*upscale_factor]

    im_sr = np.zeros(sr_size)
    count_or = 0
    count_and = 0
    for i in range(len(list_image_input_dir)):
        if i != idx_ref:
            im_to_register = rgb2gray(io.imread(list_image_input_dir[i]))

            shifted, error, diffphase = phase_cross_correlation(im_ref, im_to_register,upsample_factor=upscale_factor)
            if shifted[0] != int(shifted[0]) or shifted[1] != int(shifted[1]):
                count_or += 1
            if shifted[0] != int(shifted[0]) and shifted[1] != int(shifted[1]):
                count_and += 1
            print(i, shifted)
    print("at least 1 shift :", count_or, "| both shifts :", count_and)

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

