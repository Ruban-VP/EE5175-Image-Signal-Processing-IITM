'''
************************************************************************************
EE5175: Image signal processing - Lab 10 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 08/04/2023
Execution command: python3 EE19B138_Lab_10_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im
import math as m
import matplotlib.pyplot as plt                       # Importing the required libraries

'''
************************************************************************************
Otsu's thresholding:
************************************************************************************
'''

image = im.open('palmleaf1.png').convert("L")         # Source image opened as an 8-bit grayscale intensity matrix
img1 = np.asarray(image)                              # Converted into numpy array
dim = np.shape(img1)
xmax1 = dim[0]
ymax1 = dim[1]                                        # Image dimensions are obtained
out_img1 = np.zeros([xmax1,ymax1])

image = im.open('palmleaf2.png').convert("L")         # Source image opened as an 8-bit grayscale intensity matrix
img2 = np.asarray(image)                              # Converted into numpy array
dim = np.shape(img2)
xmax2 = dim[0]
ymax2 = dim[1]                                        # Image dimensions are obtained
out_img2 = np.zeros([xmax2,ymax2])

bins_range = [i for i in range(257)]
hist_1 = np.histogram(img1,bins_range)[0]
hist_2 = np.histogram(img2,bins_range)[0]             # Histogram of both images obtained
bins_range = bins_range[:-1]                          # Array of 8-bit gray levels

metrics_1 = [0]*257                                   # Lists to store the between-class variance values
metrics_2 = [0]*257                                   # for each image

N1 = xmax1*ymax1
N2 = xmax2*ymax2                                      # Number of pixels in each image
sum_1 = np.sum(np.multiply(hist_1,bins_range))
sum_2 = np.sum(np.multiply(hist_2,bins_range))        # Cumulative intensity sum for each image
mean_1 = sum_1/N1
mean_2 = sum_2/N2                                     # Mean intensity of each image

curr_sum_1 = 0
curr_sum_2 = 0
curr_N1 = 0
curr_N2 = 0

# For loop to find the between-class variance for each threshold value
for thr in range(255):
 
	curr_N1 = curr_N1 + hist_1[thr]                   # Class 1 size and intensity sum computed for 
	curr_sum_1 = curr_sum_1 + (thr*hist_1[thr])       # current threshold 

	# Between-class variance computed for current threshold value
	if curr_N1!=0 and curr_N1!=N1:
		temp = ((curr_sum_1)**2/(curr_N1))+((sum_1-curr_sum_1)**2/(N1-curr_N1))
		metrics_1[thr+1] = (temp/N1)-(mean_1**2)

	curr_N2 = curr_N2 + hist_2[thr]                   # Class 1 size and intensity sum computed for 
	curr_sum_2 = curr_sum_2 + (thr*hist_2[thr])       # current threshold 
	
	# Between-class variance computed for current threshold value
	if curr_N2!=0 and curr_N2!=N2:
		temp = ((curr_sum_2)**2/(curr_N2))+((sum_2-curr_sum_2)**2/(N2-curr_N2))
		metrics_2[thr+1] = (temp/N2)-(mean_2**2)

thr1 = np.argmax(metrics_1)-1                         # Otsu thresholds obtained for both the images by maximizing
thr2 = np.argmax(metrics_2)-1                         # the between-class variance values

# Otsu thresholding applied for image 1
for i in range(xmax1):
	for j in range(ymax1):

		if img1[i,j]<=thr1:
			out_img1[i,j]=0
		else:
			out_img1[i,j]=255

# Otsu thresholding applied for image 2
for i in range(xmax2):
	for j in range(ymax2):
		
		if img2[i,j]<=thr2:
			out_img2[i,j]=0
		else:
			out_img2[i,j]=255
	
data = im.fromarray(out_img1)
data = data.convert("L")
data.save('out_img1.png')                             # Target image is saved into a .png file

data = im.fromarray(out_img2)
data = data.convert("L")
data.save('out_img2.png')                             # Target image is saved into a .png file