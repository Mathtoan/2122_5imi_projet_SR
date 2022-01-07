#%% Import
import argparse
import os

from numpy import float64

from utils import *

#%% Parser
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('-d', '--device', type=str, default='iPhone13Pro',
                    help='Choose the device', choices=['iPhone13Pro'])
parser.add_argument('-s', '--scene', type=str, default='scene1',
                    help='Choose the scene', choices=['scene1'])
parser.add_argument('-f', '--upscale_factor', type=int, default='4',
                    help='Choose the upscaling factor')
parser.add_argument('-i', '--iterations', type=int, default='100',
                    help='Choose the number of iterations')
parser.add_argument('-o', '--sigma', type=float, default='0.05',
                    help='Choose the value of sigma')
parser.add_argument('-c', '--color', type=str, default='gray',
                    help='Choose the color of the output image', choices=['gray','rgb'])
parser.add_argument('-r', '--ref', type=int, default='9',
                    help='Choose the reference image')

args = parser.parse_args()

#%% Parameters
upscale_factor = args.upscale_factor
it = args.iterations
sigma = args.sigma
color = args.color
idx_ref = args.ref

print('RUNNING PARAMETER', 
      '\nUpscale factor :', upscale_factor,
      '\nNumber of iteration of Papoulis-Gerchberg :', it,
      '\nSigma :', sigma)

#%% Path
input_dir = os.path.join('fig', args.device, args.scene)
output_dir = os.path.join('output', args.device, args.scene, 'up_'+str(upscale_factor)+'_it_'+str(it)+'_sigma_'+str(sigma))

list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()


#%% Registration and SR grid creation
# for idx_ref in range(len(list_image_input_dir)):
#     im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)

im_groundtruth = io.imread(list_image_input_dir[idx_ref])
if color=='gray':
    im_groundtruth = rgb2gray(im_groundtruth)

HR_grid_txt_dir = os.path.join('output', args.device, args.scene, 'HR_grid_'+str(idx_ref)+'.txt')
# Load saved HR grid if already generated
if os.path.exists(HR_grid_txt_dir):
    print('Loading ', HR_grid_txt_dir)
    HR_grid = np.loadtxt(HR_grid_txt_dir, dtype=float)
    im_ref = rescale(rgb2gray(io.imread(list_image_input_dir[idx_ref])), 1/upscale_factor)
    print(HR_grid.shape)
else:
    im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)
    HR_grid = creation_HR_grid(im_ref, upscale_factor, im_to_register_list, registration_shifts, color)
    np.savetxt(HR_grid_txt_dir, HR_grid, fmt='%f')

if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)
io.imsave(os.path.join(output_dir,'hr_grid_'+str(idx_ref)+'.png'), HR_grid)

io.imsave(os.path.join(output_dir,'groundtruth.png'), im_groundtruth)
io.imsave(os.path.join(output_dir,'lr_image.png'), im_ref)

#%% Papoulis-Gerchberg method
im_sr = PG_method(HR_grid, im_ref, sigma, upscale_factor, it, display_filter=True)
io.imsave(os.path.join(output_dir,'sr_image_new.png'), im_sr.real)

if color=='gray':
    colmap='gray'
elif color=='rgb':
    colmap='viridis'
else:
    print('Undefined color')
    exit()

plt.figure()
plt.subplot(221)
plt.imshow(im_groundtruth, colmap)
plt.title('Groundtruth')
plt.subplot(222)
plt.imshow(im_ref, colmap)
plt.title('LR image')
plt.subplot(223)
plt.imshow(HR_grid, colmap)
plt.title('HR grid')
plt.subplot(224)
plt.imshow(im_sr.real, colmap)
plt.title('SR image')
plt.show()
plt.close()

