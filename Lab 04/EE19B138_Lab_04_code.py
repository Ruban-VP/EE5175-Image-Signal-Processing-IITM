'''
************************************************************************************
EE5175: Image signal processing - Lab 04 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 22/02/2023
Execution command: python3 EE19B138_Lab_04_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im
import math as m                                      # Importing the required libraries

'''
************************************************************************************
Space-invariant blurring:
************************************************************************************
'''

image = im.open('Mandrill.png').convert("L")          # Source image opened as an 8-bit grayscale intensity matrix
inp_img = np.asarray(image)                           # Converted into numpy array
dim = np.shape(inp_img)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained

sigmas = [1.6, 1.2, 1.0, 0.6, 0.3, 0]                 # Standard deviation values required for gaussian kernels

for count in range(len(sigmas)):                      # For loop to loop over the sigmas
 
	sigma = sigmas[count]                             # Current value of sigma

	# If standard deviation is non-zero
	if sigma!=0:
		l = m.ceil(6*sigma+1)                         # Kernel size
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

	data = im.fromarray(out_img)
	data = data.convert("L")
	data.save('img'+str(count+1)+'.png')              # Output image is saved into a .png file
