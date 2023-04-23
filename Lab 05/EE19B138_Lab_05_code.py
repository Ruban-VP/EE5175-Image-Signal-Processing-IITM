'''
************************************************************************************
EE5175: Image signal processing - Lab 05 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 26/02/2023
Execution command: python3 EE19B138_Lab_05_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im                           # Importing the required libraries

'''
************************************************************************************
Space-variant blurring:
************************************************************************************
'''

####################################################################################
# Blurring Globe.png
####################################################################################

image = im.open('Globe.png').convert("L")             # Source image opened as an 8-bit grayscale intensity matrix
inp_img = np.asarray(image)                           # Converted into numpy array
dim = np.shape(inp_img)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained
N = xmax

out_img = np.zeros([xmax,ymax])

A = 2.0
B = (N*N)/(2*np.log(200))                             # A,B values are computed based on the given info

for m in range(xmax):
	for n in range(ymax):

		arg = (((m-(N/2))**2)+((n-(N/2))**2))/B
		sigma = A*np.exp(-arg)                        # Sigma for current pixel computed

		l = int(np.ceil(6*sigma+1))             
		mid = (l-1)/2                                 # Kernel length for current pixel

		xlimit = min(m+1,l)
		ylimit = min(n+1,l)                           # Convolution limits

		# For loop to generate the gaussian kernel
		kernel = np.zeros([l,l])
		if sigma!=0:
			for i in range(l):
				for j in range(l):
					arg = (20/sigma)-(((i-mid)**2)+((j-mid)**2))/(2*sigma*sigma)
					kernel[i,j] = np.exp(arg)
			kernel = kernel/np.sum(kernel)            # Kernel normalization	
		else:
			kernel[0,0] = 1                           # If sigma is zero, kernel is simply 1

		# For loop to assign output image intensities 
		for i in range(xlimit):
			for j in range(ylimit):
				out_img[m,n] = out_img[m,n] + (kernel[i,j]*inp_img[m-i,n-j])               


data = im.fromarray(out_img)
data = data.convert("L")
data.save('Globe_blur.png')                           # Blurred globe image saved in a .png file           

####################################################################################
# Blurring Nautilus.png
####################################################################################

image = im.open('Nautilus.png').convert("L")          # Source image opened as an 8-bit grayscale intensity matrix
inp_img = np.asarray(image)                           # Converted into numpy array
dim = np.shape(inp_img)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained

## Linear space-invariant blurring code:

sigmas = [1]                 						  # Standard deviation values required for gaussian kernels
for count in range(len(sigmas)):                      # For loop to loop over the sigmas
 
	sigma = sigmas[count]                             # Current value of sigma

	# If standard deviation is non-zero
	if sigma!=0:
		l = int(np.ceil(6*sigma+1))                   # Kernel size
		mid = (l-1)/2                                 # Kernel center
		kernel = np.zeros([l,l])

		# For loop to generate the gaussian kernel
		for i in range(l):
			for j in range(l):
				arg = (((i-mid)**2)+((j-mid)**2))/(2*sigma*sigma)
				kernel[i,j] = np.exp(-arg)

		kernel = kernel/np.sum(kernel)                # Normalizing the kernel to get kernel sum as unity     
		out_img = np.zeros([xmax+l-1,ymax+l-1])       # Output image array initialized

		# Input image array is zero padded to perform the 2-D convolution at image edges properly
		inp_img_ext = np.concatenate((np.zeros([xmax,l-1]),inp_img,np.zeros([xmax,l-1])),axis=1)
		inp_img_ext = np.concatenate((np.zeros([l-1,ymax+2*l-2]),inp_img_ext,np.zeros([l-1,ymax+2*l-2])),axis=0)

		# For loop to perform the convolution for different pixel locations 
		for i in range(xmax+l-1):
			for j in range(ymax+l-1):
				img_slice = inp_img_ext[i:i+l,j:j+l]
				out_img[i,j] = np.sum(np.multiply(img_slice,kernel))

	# If standard deviation is zero, there is no blurring and hence, output image is same as input image
	else:
		out_img = inp_img
out_img1 = out_img[:xmax,:ymax]

## Linear space-variant blurring code:

out_img = np.zeros([xmax,ymax])
for m in range(xmax):
	for n in range(ymax):

		sigma = 1                                     # Sigma for current pixel computed
		l = int(np.ceil(6*sigma+1))             
		mid = (l-1)/2                                 # Kernel length for current pixel

		xlimit = min(m+1,l)
		ylimit = min(n+1,l)                           # Convolution limits

		# For loop to generate the gaussian kernel
		kernel = np.zeros([l,l])
		if sigma!=0:
			for i in range(l):
				for j in range(l):
					arg = (((i-mid)**2)+((j-mid)**2))/(2*sigma*sigma)
					kernel[i,j] = np.exp(-arg)
			kernel = kernel/np.sum(kernel)            # Kernel normalization
		else:
			kernel[0,0] = 1                           # If sigma is zero, kernel is simply 1

		# For loop to assign output image intensities 
		for i in range(xlimit):
			for j in range(ylimit):
				out_img[m,n] = out_img[m,n] + (kernel[i,j]*inp_img[m-i,n-j])    

out_img2 = out_img
diff = np.abs(out_img1-out_img2)                      # Difference in intensities between the two outputs
mean_diff = np.mean(diff)

print("The mean difference in the output intensities is: "+str(mean_diff))

data = im.fromarray(out_img1)
data = data.convert("L")
data.save('Nautilus_LSI_blur.png')                    # LSI blurred Nautilus image saved in a .png file  

data = im.fromarray(out_img2)
data = data.convert("L")
data.save('Nautilus_LSV_blur.png')                    # LSV blurred Nautilus image saved in a .png file  