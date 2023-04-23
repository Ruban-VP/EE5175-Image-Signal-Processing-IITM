'''
************************************************************************************
EE5175: Image signal processing - Lab 12 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 20/04/2023
Execution command: python3 EE19B138_Lab_12_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import cv2
import numpy as np
from PIL import Image as im
import math as m
import matplotlib.pyplot as plt                       # Importing the required libraries

'''
************************************************************************************
Non-local means filtering (NLM):
************************************************************************************
'''

image = im.open('krishna_0_001.png')                  # Source image opened as an 24-bit RGB intensity matrix
g = np.asarray(image)                                 # Converted into numpy array
g = g/255
dim = np.shape(g)

image = im.open('krishna.png')                        # Source image opened as an 24-bit RGB intensity matrix
f = np.asarray(image)                                 # Converted into numpy array
f = f/255
dim = np.shape(f)

xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained

base_MSE = (np.linalg.norm(g-f)**2)/(xmax*ymax*3)
base_PSNR = 10*m.log(1/base_MSE,10)                   # Baseline PSNR is calculated
print("Baseline PSNR is:",base_PSNR)

W_vals = [3,5]
W_sim = 3
std_vals = np.array([0.1,0.2,0.3,0.4,0.5])
var_vals = std_vals**2
W_len = len(W_vals)
var_len = len(var_vals)                               # Parameters of the assignment 

PSNR_vals_NLM = np.zeros([W_len,var_len])
PSNR_vals_gauss = np.zeros(var_len)                   # PSNR arrays

x1 = 30
y1 = 45
x2 = 37
y2 = 57                                               # Pixel locations
wl = 5

patch_noisy_1 = g[x1-wl:x1+wl+1,y1-wl:y1+wl+1,:]
patch_noisy_2 = g[x2-wl:x2+wl+1,y2-wl:y2+wl+1,:]      # Noisy patches obtained

### NLM filtering:
for W_ind in range(W_len):
	W = W_vals[W_ind]
	gap = W_sim+W
	f_sub = f[gap:xmax-gap,gap:ymax-gap,:]            # Sub-image for which denoising is done

	dim_sub = np.shape(f_sub)
	xsub = dim_sub[0]
	ysub = dim_sub[1]                                 # Dimensions of sub-image obtained
 
	f_hat = np.zeros(dim_sub)
	SW_xlen = xmax-(2*W_sim)
	SW_ylen = ymax-(2*W_sim)
	sim_windows = np.zeros([2*W_sim+1,2*W_sim+1,3,SW_xlen,SW_ylen])  

	# For loop to obtain and store similarity windows for all pixels
	for i in range(SW_xlen):
		for j in range(SW_ylen):
			curr_loc_x = W_sim+i
			curr_loc_y = W_sim+j
			sim_windows[:,:,:,i,j] = g[curr_loc_x-W_sim:curr_loc_x+W_sim+1,curr_loc_y-W_sim:curr_loc_y+W_sim+1,:]

	for var_ind in range(var_len):
		var = var_vals[var_ind]

		for i in range(xsub):
			for j in range(ysub):

				curr_loc_x = gap+i
				curr_loc_y = gap+j

				# Similarity and search windows obtained for current pixel 
				curr_window = g[curr_loc_x-W:curr_loc_x+W+1,curr_loc_y-W:curr_loc_y+W+1,:]
				curr_sim_window = g[curr_loc_x-W_sim:curr_loc_x+W_sim+1,curr_loc_y-W_sim:curr_loc_y+W_sim+1,:]
				curr_sim_window = curr_sim_window[...,np.newaxis,np.newaxis]

				# Similarity windows of all pixels present in the search window are obtained
				curr_sim_windows = sim_windows[:,:,:,i:i+2*W+1,j:j+2*W+1]
				sim_window_diff = curr_sim_windows-curr_sim_window

				# Weights are computed according to the formula given in the assignment
				temp = np.linalg.norm(sim_window_diff, axis=0)**2
				temp = np.sum(temp, axis=0)
				temp = np.sum(temp, axis=0)
				
				curr_weights = np.exp(-temp/var)
				curr_weights = curr_weights[...,np.newaxis]
				curr_weights = curr_weights/np.sum(curr_weights)

				out_window = curr_window*curr_weights
				temp = np.sum(out_window, axis=0)
				f_hat[i,j,:] = np.sum(temp, axis=0)   # Output intensity is computed 

				if W==5 and var==0.25:
					if curr_loc_x==x1 and curr_loc_y==y1:
						kernel_Q3_1 = curr_weights[:,:,0]
					elif curr_loc_x==x2 and curr_loc_y==y2:
						kernel_Q3_2 = curr_weights[:,:,0]

		if W==5 and var==0.25:
			patch_NLM_1 = f_hat[x1-gap-wl:x1-gap+wl+1,y1-gap-wl:y1-gap+wl+1,:]
			patch_NLM_2 = f_hat[x2-gap-wl:x2-gap+wl+1,y2-gap-wl:y2-gap+wl+1,:]

		MSE = (np.linalg.norm(f_hat-f_sub)**2)/(xsub*ysub*3)
		PSNR_vals_NLM[W_ind,var_ind] = 10*m.log(1/MSE,10)   # PSNR computed and stored

### Gaussian filtering:
for var_ind in range(var_len+1):  

	if var_ind!=var_len:                    
		var = var_vals[var_ind]                            
		l = 7 
	else:
		var = 1
		l = 11 
	                       
	mid = int((l-1)/2)                               
	kernel = np.zeros([l,l,1])

	# For loop to generate the gaussian kernel
	for i in range(l):
		for j in range(l):
			arg = (((i-mid)**2)+((j-mid)**2))/(2*var)
			kernel[i,j,0] = np.exp(-arg)

	kernel = kernel/np.sum(kernel)                    # Normalizing the kernel to get kernel sum as unity 

	if var_ind==var_len:
		kernel_Q3_3 = kernel[:,:,0]
		kernel_Q3_4 = kernel[:,:,0]

	f_hat = np.zeros([xmax+l-1,ymax+l-1,3])           # Output image array initialized     

	# Input image array is zero padded to perform the 2-D convolution at image edges properly
	g_ext = np.concatenate((np.zeros([xmax,l-1,3]),g,np.zeros([xmax,l-1,3])),axis=1)
	g_ext = np.concatenate((np.zeros([l-1,ymax+2*l-2,3]),g_ext,np.zeros([l-1,ymax+2*l-2,3])),axis=0)

	# For loop to perform the convolution for different pixel locations 
	for i in range(xmax+l-1):
		for j in range(ymax+l-1):
			img_slice = g_ext[i:i+l,j:j+l,:]
			temp = np.sum(img_slice*kernel, axis=0)
			f_hat[i,j,:] = np.sum(temp, axis=0)

	# Edges are cut and output image of relevant dimensions obtained
	f_hat = f_hat[mid:xmax+l-mid-1,mid:ymax+l-mid-1,:]  

	if var_ind==var_len:
		patch_gauss_1 = f_hat[x1-wl:x1+wl+1,y1-wl:y1+wl+1,:]
		patch_gauss_2 = f_hat[x2-wl:x2+wl+1,y2-wl:y2+wl+1,:]

	MSE = (np.linalg.norm(f_hat-f)**2)/(xmax*ymax*3)

	if var_ind!=var_len:
		PSNR_vals_gauss[var_ind] = 10*m.log(1/MSE,10) # PSNR computed and stored	

# PSNR values are plotted against the standard deviation values
plt.figure(1)
plt.plot(std_vals,PSNR_vals_NLM[0,:])
plt.plot(std_vals,PSNR_vals_NLM[1,:])
plt.plot(std_vals,PSNR_vals_gauss)
plt.xlabel(r'$\sigma$')
plt.ylabel('PSNR (in dB)')
plt.title(r'$\sigma$ vs PSNR plot')
plt.legend(['W=3','W=5','Gauss'])
plt.savefig('PSNR.png', bbox_inches='tight')

# Kernels are expressed in logarithmic scale
kernel_Q3_1 = np.log(kernel_Q3_1)
kernel_Q3_2 = np.log(kernel_Q3_2)
kernel_Q3_3 = np.log(kernel_Q3_3)
kernel_Q3_4 = np.log(kernel_Q3_4)

### Kernels and image patches are plotted and saved as .png files 

plt.figure(2)
plt.imshow(kernel_Q3_1,cmap='gray')                        
plt.title('NLM kernel 1')
plt.savefig('NLM_kernel_1.png', bbox_inches='tight')

plt.figure(3)
plt.imshow(kernel_Q3_2,cmap='gray')                        
plt.title('NLM kernel 2')
plt.savefig('NLM_kernel_2.png', bbox_inches='tight')

plt.figure(4)
plt.imshow(kernel_Q3_3,cmap='gray')                        
plt.title('Gaussian kernel 1')
plt.savefig('Gaussian_kernel_1.png', bbox_inches='tight')

plt.figure(5)
plt.imshow(kernel_Q3_4,cmap='gray')                        
plt.title('Gaussian kernel 2')
plt.savefig('Gaussian_kernel_2.png', bbox_inches='tight')

plt.figure(6)
plt.imshow(patch_noisy_1)                        
plt.title('Noisy patch 1')
plt.savefig('Noisy_patch_1.png', bbox_inches='tight')

plt.figure(7)
plt.imshow(patch_noisy_2)                        
plt.title('Noisy patch 2')
plt.savefig('Noisy_Patch_2.png', bbox_inches='tight')

plt.figure(8)
plt.imshow(patch_NLM_1)                        
plt.title('NLM filtered patch 1')
plt.savefig('NLM_filtered_patch_1.png', bbox_inches='tight')

plt.figure(9)
plt.imshow(patch_NLM_2)                        
plt.title('NLM filtered patch 2')
plt.savefig('NLM_filtered_patch_2.png', bbox_inches='tight')

plt.figure(10)
plt.imshow(patch_gauss_1)                        
plt.title('Gaussian filtered patch 1')
plt.savefig('Gaussian_filtered_patch_1.png', bbox_inches='tight')

plt.figure(11)
plt.imshow(patch_gauss_2)                        
plt.title('Gaussian filtered patch 2')
plt.savefig('Gaussian_filtered_patch_2.png', bbox_inches='tight')

plt.show()