import argparse
import os

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

print('RUNNING PARAMETER', 
      '\nUpscale factor :', upscale_factor,
      '\nNumber of iteration of Papoulis-Gerchberg :', it)

# Path
input_dir = os.path.join('fig', args.device, args.scene)
output_dir = os.path.join('output', args.device, args.scene, 'up_'+str(upscale_factor)+'_it_'+str(it)+'_sigma_'+str(sigma))

list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()


# Registration and SR grid creation
# for idx_ref in range(len(list_image_input_dir)):
#     im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)

idx_ref = 9
im_groundtruth = rgb2gray(io.imread(list_image_input_dir[idx_ref]))
im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)

HR_grid = creation_HR_grid(im_ref, upscale_factor, im_to_register_list, registration_shifts, color)

if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)
io.imsave(os.path.join(output_dir,'groundtruth.png'), im_groundtruth)
io.imsave(os.path.join(output_dir,'lr_image.png'), im_ref)
io.imsave(os.path.join(output_dir,'hr_grid.png'), HR_grid)

im_sr = PG_method(HR_grid, im_ref, sigma, upscale_factor, it)
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
plt.imshow(im_groundtruth, colmap)
plt.title('Groundtruth')
plt.subplot(222)
plt.imshow(im_ref, colmap)
plt.title('LR image')
plt.subplot(223)
plt.imshow(HR_grid, colmap)
plt.title('HR grid')
plt.subplot(224)
plt.imshow(im_sr, colmap)
plt.title('SR image')
plt.show()
plt.close()

