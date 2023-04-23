'''
************************************************************************************
EE5175: Image signal processing - Lab 13 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 22/04/2023
Execution command: python3 EE19B138_Lab_13_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im
import math as m                                      # Importing the required libraries

'''
************************************************************************************
2D DFT and IDFT functions:
************************************************************************************
'''

def two_D_DFT(inp,M,N):

	inp = np.fft.ifftshift(inp, axes=(0,))            # Periodic shifting to make image matrix 
	inp = np.fft.ifftshift(inp, axes=(1,))            # upper left corner as origin

	out = np.zeros([M,N], dtype=complex)
	for m in range(M):
		out[m,:] = np.fft.fft(inp[m,:])               # 1D FFT taken for each row

	for l in range(N):
		out[:,l] = np.fft.fft(out[:,l])               # 1D FFT taken for each column

	out = np.fft.fftshift(out, axes=(0,))
	out = np.fft.fftshift(out, axes=(1,))             # Periodic shifting to make DFT matrix center as origin

	return out

def two_D_IDFT(inp,M,N):

	inp = np.fft.ifftshift(inp, axes=(0,))            # Periodic shifting to make DFT matrix
	inp = np.fft.ifftshift(inp, axes=(1,))            # upper left corner as origin

	out = np.zeros([M,N], dtype=complex)
	for k in range(M):
		out[k,:] = np.fft.ifft(inp[k,:])              # 1D IFFT taken for each row           

	for n in range(N):
		out[:,n] = np.fft.ifft(out[:,n])              # 1D IFFT taken for each column

	out = np.fft.fftshift(out, axes=(0,))
	out = np.fft.fftshift(out, axes=(1,))             # Periodic shifting to make image matrix center as origin

	return out

'''
************************************************************************************
Wiener filtering:
************************************************************************************
'''

image = im.open('lena.png').convert("L")              # Source image opened as an 8-bit grayscale intensity matrix
inp_img = np.asarray(image)                           # Converted into numpy array
dim = np.shape(inp_img)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained

sigma_n_vals = [1,5,15]                               # Standard deviation values for the Gaussian noise
sigma_b = 1.5                                         # Standard deviation values required for Gaussian kernels

k = np.arange(0.01,2,0.001)                           # 'k' values
k_vals = np.zeros(len(sigma_n_vals))
out_imgs = np.zeros([xmax,ymax,len(sigma_n_vals)])

l = m.ceil(6*sigma_b+1)                               # Kernel size
if l%2==0:
	l = l+1

mid = int((l-1)/2)                                    # Kernel center
kernel = np.zeros([l,l])

### Gaussian blurring snippet:
# For loop to generate the Gaussian kernel
for i in range(l):
	for j in range(l):
		arg = (((i-mid)**2)+((j-mid)**2))/(2*sigma_b*sigma_b)
		kernel[i,j] = np.exp(-arg)

kernel = kernel/np.sum(kernel)                        # Normalizing the kernel to get kernel sum as unity     
blur_img = np.zeros([xmax+l-1,ymax+l-1])              # Output image array initialized

# Input image array is zero padded to perform the 2-D convolution at image edges properly
inp_img_ext = np.concatenate((np.zeros([xmax,l-1]),inp_img,np.zeros([xmax,l-1])),axis=1)
inp_img_ext = np.concatenate((np.zeros([l-1,ymax+2*l-2]),inp_img_ext,np.zeros([l-1,ymax+2*l-2])),axis=0)

# For loop to perform the convolution for different pixel locations 
for i in range(xmax+l-1):
	for j in range(ymax+l-1):
		img_slice = inp_img_ext[i:i+l,j:j+l]
		blur_img[i,j] = np.sum(np.multiply(img_slice,kernel))

blur_img = blur_img[mid:xmax+l-mid-1,mid:ymax+l-mid-1]   # Blurred image obtained

y_ext = int((ymax-l)/2)
x_ext = int((xmax-l)/2)

# Kernel zero-padded for DFT computation
kernel_ext = np.concatenate((np.zeros([l,y_ext]),kernel,np.zeros([l,y_ext])),axis=1)
kernel_ext = np.concatenate((np.zeros([x_ext,ymax]),kernel_ext,np.zeros([x_ext,ymax])),axis=0)

kernel_DFT = two_D_DFT(kernel_ext,xmax,ymax)
kernel_conj_DFT = np.conjugate(kernel_DFT)            # DFT computation for the kernel             

### Weiner filtering snippet:
# For loop to loop over every noise variance value
for sigma_ind in range(len(sigma_n_vals)):
	sigma_n = sigma_n_vals[sigma_ind]
	noisy_img = blur_img + (sigma_n*np.random.randn(xmax,ymax))  # Noisy degraded image generated
	
	out_img_arrs = np.zeros([xmax,ymax,len(k)])
	RMSE_vals = np.zeros(len(k))

	noisy_img_DFT = two_D_DFT(noisy_img,xmax,ymax)               # DFT of noisy image

	# For loop to loop over every 'k' value
	for ind in range(len(k)):
		val = k[ind]
		den = np.multiply(kernel_DFT,kernel_conj_DFT)+val
		wein_DFT = np.divide(kernel_conj_DFT,den)
		out_DFT = np.multiply(wein_DFT,noisy_img_DFT)            # Weiner filtered output image DFT

		out_img = np.abs(two_D_IDFT(out_DFT,xmax,ymax))          # Weiner filtered output image
		out_img_arrs[:,:,ind] = out_img
		RMSE_vals[ind] = np.linalg.norm(out_img-inp_img)         # RMSE calculated for current 'k'

	# For current noise variance, 'k' giving the least RMSE is found
	k_vals[sigma_ind] = k[np.argmin(RMSE_vals)]                  
	out_imgs[:,:,sigma_ind] = out_img_arrs[:,:,np.argmin(RMSE_vals)]

# For different values of noise variance, the 'k' giving least RMSE is printed
print("The k values giving min. RMSE for different noise variances are:",k_vals)

for sigma_ind in range(len(sigma_n_vals)):
	data = im.fromarray(out_imgs[:,:,sigma_ind])
	data = data.convert("L")
	data.save('out_img'+str(sigma_ind+1)+'.png')              # Output image is saved into a .png file