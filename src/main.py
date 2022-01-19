#%% Import
import argparse
import os
import pickle

from utils import *

#%% Parser
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('-d','--display', action='store_true',
                    help='Display the images')
parser.add_argument('-D', '--device', type=str, default='iPhone13Pro',
                    help='Choose the device', choices=['iPhone13Pro'])
parser.add_argument('-s', '--scene', type=str, default='scene1',
                    help='Choose the scene', choices=['scene1','scene2', 'scene3'])
parser.add_argument('-u', '--upscale_factor', type=int, default='4',
                    help='Choose the upscaling factor')
parser.add_argument('-f', '--filter_type', type=str, default='centered_circle',
                    help='Choose the filter')
parser.add_argument('-o', '--sigma', type=float, default='0.05',
                    help='Choose the value of sigma')
parser.add_argument('-c', '--color', type=str, default='gray',
                    help='Choose the color of the output image', choices=['gray','rgb'])
parser.add_argument('-r', '--ref', type=int, default='0',
                    help='Choose the reference image')
parser.add_argument('-m', '--method', type=str, default='POI',
                    help='Choose the registration method', choices=['translation','POI','pixel','itk','handmade'])
parser.add_argument('-i', '--intermediary_step', type=float,
                    help='Save PG method n steps')
parser.add_argument('-j','--debug', action='store_true',
                    help='Debug')
parser.add_argument('-p','--psf', action='store_true',
                    help='PSF')



args = parser.parse_args()

#%% Parameters
display = args.display
upscale_factor = args.upscale_factor
sigma = args.sigma
color = args.color
idx_ref = args.ref
method = args.method
intermediary_step = args.intermediary_step
debug = args.debug
psf = args.psf
filter_type = args.filter_type

if color=='gray':
    colmap='gray'
elif color=='rgb':
    colmap='viridis'
else:
    print('Undefined color')
    exit()

print('RUNNING PARAMETER', 
      '\nUpscale factor :', upscale_factor,
      '\nSigma :', sigma,
      '\nFilter :', filter_type)

#%% Path
input_dir = os.path.join('fig', args.device, args.scene)
if debug:
    if psf:
        output_dir = os.path.join('debug', args.device, args.scene, 'psf')
    else:
        output_dir = os.path.join('debug', args.device, args.scene)
else:
    output_dir = os.path.join('output_'+color, args.device, args.scene)
o_up_dir = os.path.join(output_dir, 'up_'+str(upscale_factor))
o_sigma_dir = os.path.join(o_up_dir, 'sigma_'+str(sigma))

if not(os.path.exists(o_sigma_dir)):
    os.makedirs(o_sigma_dir)
if not(os.path.exists('output/hist')):
    os.makedirs('output/hist')

list_image_input_dir = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if not i.startswith('.')]
list_image_input_dir.sort()


#%% Registration and SR grid creation
# for idx_ref in range(len(list_image_input_dir)):
#     im_ref, im_to_register_list, registration_shifts = computing_regitration(list_image_input_dir, idx_ref, upscale_factor)

im_groundtruth = io.imread(list_image_input_dir[idx_ref])
if color=='gray':
    try: 
        im_groundtruth = rgb2gray(im_groundtruth)
        im_groundtruth = im_groundtruth[:,:,np.newaxis]
    except: pass
im_ref = rescale(im_groundtruth, 1/upscale_factor, channel_axis=2)

HR_grid_txt_dir = os.path.join(o_up_dir, 'HR_grid_'+str(idx_ref)+'.pkl')

# Load saved HR grid if already generated
if os.path.exists(HR_grid_txt_dir):
    print('Loading ', HR_grid_txt_dir)
    # HR_grid = np.loadtxt(HR_grid_txt_dir, dtype=float)
    pkl_file = open(HR_grid_txt_dir, 'rb')
    HR_grid = pickle.load(pkl_file)
else:
    HR_grid = creation_HR_grid(im_ref, list_image_input_dir, idx_ref, upscale_factor, method, color)
    try: plt.imsave(os.path.join(o_up_dir,'hr_grid_'+str(idx_ref)+'.png'), float64_to_uint8(HR_grid), cmap=colmap)
    except: plt.imsave(os.path.join(o_up_dir,'hr_grid_'+str(idx_ref)+'.png'), float64_to_uint8(HR_grid[:,:,0]), cmap=colmap)
    # np.savetxt(HR_grid_txt_dir, HR_grid[:,:,0], fmt='%f')
    output = open(HR_grid_txt_dir, 'wb')
    pickle.dump(HR_grid, output)
    output.close()

image_histogram(HR_grid[:,:,0], 'Histogram HR grid', save_dir=os.path.join(o_up_dir,'hist_HR_grid.png'))
image_histogram(HR_grid[:,:,0], 'Histogram HR grid without 0', save_dir=os.path.join(o_up_dir,'hist_HR_grid_1.png'), bins=np.linspace(1/255,1,255))

save_im(os.path.join(o_up_dir,'groundtruth.png'), im_groundtruth, colmap, new=True)
save_im(os.path.join(o_up_dir,'lr_image_'+str(idx_ref)+'.png'), im_ref, colmap, new=True)
# exit()

#%% Papoulis-Gerchberg method
# image_histogram(im_groundtruth, 'Histogram groundtruth', save_dir=os.path.join(o_up_dir,'hist_gt.png'))
if debug:
    im_sr,H = PG_method(HR_grid,
                        save_dir=o_sigma_dir, out_filter=True, intermediary_step=intermediary_step,
                        plot_debug_intensity=True, filter_type=filter_type)
else:
    im_sr,H = PG_method(HR_grid, sigma,
                        save_dir=o_sigma_dir, out_filter=True, intermediary_step=intermediary_step, plot_debug_intensity=True,
                        filter_type=filter_type)
io.imsave(os.path.join(o_sigma_dir, 'filter.png'), float64_to_uint8(H))

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

