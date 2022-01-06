import argparse
import os

import matplotlib.pyplot as plt

from utils import *

# Parser
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('-d', '--device', type=str, default='iPhone13Pro',
                    help='Choose the device', choices=['iPhone13Pro'])
parser.add_argument('-s', '--scene', type=str, default='scene1',
                    help='Choose the scene', choices=['scene1'])
parser.add_argument('-f', '--upscale_factor', type=int, default='4',
                    help='Choose the upscaling factor')
parser.add_argument('-i', '--iterations', type=int, default='100',
                    help='Choose the number of iterations')
parser.add_argument('-o', '--sigma', type=float, default='0.4',
                    help='Choose the value of sigma')
parser.add_argument('-c', '--color', type=str, default='grey',
                    help='Choose the color of the output image', choices=['grey','rgb'])

args = parser.parse_args()

# Parameters
upscale_factor = args.upscale_factor
it = args.iterations
sigma = args.sigma
color = args.color

# Path
input_dir = os.path.join('fig', args.device, args.scene)
output_dir = os.path.join('output', args.device, args.scene, 'up_'+str(upscale_factor)+'_it_'+str(it)+'_sigma_'+str(sigma))

list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()


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

if color=='grey':
    im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1)
    im_dg = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/upscale_factor)
elif color=='rgb':
    im_ref = rescale(io.imread(list_image_input_dir[idx_ref]), 1)
    im_dg = rescale(io.imread(list_image_input_dir[idx_ref]), 1/upscale_factor, channel_axis=2)
else:
    print('Undefined color')
    exit()

shift = []
to_shift = []
count_val = 0
count_or = 0
for i in range(len(list_image_input_dir)):
    if i != idx_ref:
        if color=='grey':
            im_to_register = rescale(rgb2gray(io.imread(list_image_input_dir[i])), 1/upscale_factor)
        elif color=='rgb':
            im_to_register = rescale(io.imread(list_image_input_dir[i]), 1/upscale_factor, channel_axis=2)
        else:
            print('Undefined color')
            exit()

        shifted, error, diffphase = phase_cross_correlation(im_dg, im_to_register,upsample_factor=upscale_factor)
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

HR_grid = creation_HR_grid(im_dg, upscale_factor, to_shift, shift, color)

if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)
io.imsave(os.path.join(output_dir,'groundtruth.png'), im_ref)
io.imsave(os.path.join(output_dir,'lr_image.png'), im_dg)
io.imsave(os.path.join(output_dir,'hr_grid.png'), HR_grid)

im_sr = PG_method(HR_grid, im_dg, sigma, upscale_factor, it)
io.imsave(os.path.join(output_dir,'sr_image.png'), im_sr)

if color=='grey':
    colmap='grey'
elif color=='rgb':
    colmap='viridis'
else:
    print('Undefined color')
    exit()

plt.figure()
plt.subplot(221)
plt.imshow(im_ref, colmap)
plt.title('Groundtruth')
plt.subplot(222)
plt.imshow(im_dg, colmap)
plt.title('LR image')
plt.subplot(223)
plt.imshow(HR_grid, colmap)
plt.title('HR grid')
plt.subplot(224)
plt.imshow(im_sr, colmap)
plt.title('SR image')
plt.show()
plt.close()

