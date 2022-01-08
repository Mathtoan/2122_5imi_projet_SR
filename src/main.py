#%% Import
import argparse
import os

from numpy import float64

from utils import *

#%% Parser
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('-d','--display', action='store_true',
                    help='Display the images')
parser.add_argument('-D', '--device', type=str, default='iPhone13Pro',
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
parser.add_argument('-S','--savesteps', action='store_true',
                    help='Save PG method step every 100')



args = parser.parse_args()

#%% Parameters
display = args.display
upscale_factor = args.upscale_factor
it = args.iterations
sigma = args.sigma
color = args.color
idx_ref = args.ref
savesteps = args.savesteps

print('RUNNING PARAMETER', 
      '\nUpscale factor :', upscale_factor,
      '\nNumber of iteration of Papoulis-Gerchberg :', it,
      '\nSigma :', sigma)

#%% Path
input_dir = os.path.join('fig', args.device, args.scene)
output_dir = os.path.join('output', args.device, args.scene)
o_up_dir = os.path.join(output_dir, 'up_'+str(upscale_factor))
o_sigma_dir = os.path.join(o_up_dir, 'sigma_'+str(sigma))
o_it_dir = os.path.join(o_sigma_dir, 'it_'+str(it))

if not(os.path.exists(o_it_dir)):
    os.makedirs(o_it_dir)

list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()


#%% Registration and SR grid creation
# for idx_ref in range(len(list_image_input_dir)):
#     im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)

im_groundtruth = io.imread(list_image_input_dir[idx_ref])
if color=='gray':
    im_groundtruth = rgb2gray(im_groundtruth)

HR_grid_txt_dir = os.path.join(o_up_dir, 'HR_grid_'+str(idx_ref)+'.txt')

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

save_im_new(os.path.join(o_up_dir,'hr_grid_'+str(idx_ref)+'.png'), HR_grid)
save_im_new(os.path.join(o_up_dir,'groundtruth.png'), im_groundtruth)
save_im_new(os.path.join(o_up_dir,'lr_image_'+str(idx_ref)+'.png'), im_ref)

#%% Papoulis-Gerchberg method
if not(savesteps):
    im_sr,H = PG_method(HR_grid, im_ref, sigma, upscale_factor, it, out_filter=True)
else:
    im_sr,H = PG_method(HR_grid, im_ref, sigma, upscale_factor, it, out_filter=True, save_every=True, save_dir=o_sigma_dir)
io.imsave(os.path.join(o_sigma_dir, 'filter.png'), H)
io.imsave(os.path.join(o_it_dir,'sr_image_new.png'), im_sr.real)

if color=='gray':
    colmap='gray'
elif color=='rgb':
    colmap='viridis'
else:
    print('Undefined color')
    exit()

if display:
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

