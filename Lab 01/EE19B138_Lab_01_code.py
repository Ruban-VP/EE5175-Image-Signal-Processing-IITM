'''
************************************************************************************
EE5175: Image signal processing - Lab 01 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 02/02/2023
Execution command: python3 EE19B138_Lab_01_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im
import math as m
import matplotlib.pyplot as plt      # Importing the required libraries

'''
************************************************************************************
Bilinear interpolator function:
************************************************************************************
'''

def Bil_interp(x,y,a,b,I1,I2,I3,I4):
	out_int = ((1-a)*(1-b)*I1)+((1-a)*b*I2)+(a*(1-b)*I3)+(a*b*I4)  # Bilinear interpolation formula
	return out_int

'''
************************************************************************************
Image translation:
************************************************************************************
'''

image = im.open('lena_translate.png').convert("L")    # Source image opened as an 8-bit grayscale intensity matrix
in_arr = np.asarray(image)                            # Converted into numpy array
dim = np.shape(in_arr)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained

plt.figure(1)
plt.imshow(in_arr,cmap='gray')                        # Image matrix is plotted 
plt.title('Lena')

tx = 3.75
ty = 4.3                                              # Translation offsets

out_arr = np.zeros(dim)                               # Target image matrix initialized with zeros

# For loop to run over each co-ordinate of the target image
for i in range(xmax):
	for j in range(ymax):

		x = i-tx
		y = j-ty                                      # Inverse of translation (Target-to-source mapping) is performed 
		xlow = m.floor(x)
		ylow = m.floor(y)
		a = x-xlow
		b = y-ylow                                    # a,b values are calculated

		# If the back calculated source image co-ordinates lie outside the matrix, intensity zero is assigned (Dark spot)
		if x<0 or x>xmax-1 or y<0 or y>ymax-1:
			out_arr[i,j] = 0                           
		else:
			# Intensities of the co-ordinates of the square that contain the back calculated source point
			I1 = in_arr[xlow,ylow] 
			I2 = in_arr[xlow,ylow+1]
			I3 = in_arr[xlow+1,ylow]
			I4 = in_arr[xlow+1,ylow+1]                
			temp = Bil_interp(x,y,a,b,I1,I2,I3,I4)    # Bilinear interpolator function is invoked to obtain the target intensity
			if temp<0:
				out_arr[i,j] = 0
			elif temp>255:
				out_arr[i,j] = 255
			else:
				out_arr[i,j] = round(temp)            # Intensities are rounded to the nearest integer

plt.figure(2)
plt.imshow(out_arr,cmap='gray')                       # Target image is plotted
plt.title('Translated Lena')

data = im.fromarray(out_arr)
data = data.convert("L")
data.save('lena_translated.png')                      # Target image is saved into a .png file

'''
************************************************************************************
Image rotation:
************************************************************************************
'''

image = im.open('pisa_rotate.png').convert("L")       # Source image opened as an 8-bit grayscale intensity matrix
in_arr = np.asarray(image)                            # Converted into numpy array
dim = np.shape(in_arr)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained
xmid = (xmax-1)/2
ymid = (ymax-1)/2                                     # Image center co-ordinates are found

plt.figure(3)
plt.imshow(in_arr,cmap='gray')                        # Image matrix is plotted 
plt.title('Pisa')

theta_deg = 4
theta_rad = theta_deg*m.pi/180                        # Rotation angle (Clockwise is positive)

out_arr = np.zeros(dim)                               # Target image matrix initialized with zeros

# For loop to run over each co-ordinate of the target image
for i in range(xmax):
	for j in range(ymax):
		
		# Target co-ordinates w.r.t image center are found
		i_cent_ref = i-xmid
		j_cent_ref = j-ymid

		# Target-to-source mapping
		# Inverse of rotation (Anti-clockwise roation) operation is performed 
		x_cent_ref = (m.cos(theta_rad)*i_cent_ref)+(m.sin(theta_rad)*j_cent_ref)
		y_cent_ref = -(m.sin(theta_rad)*i_cent_ref)+(m.cos(theta_rad)*j_cent_ref)

		# Source co-ordinates are found w.r.t the matrix origin (Top left) are found 
		x = xmid+x_cent_ref
		y = ymid+y_cent_ref                           

		xlow = m.floor(x)
		ylow = m.floor(y)
		a = x-xlow
		b = y-ylow                                    # a,b values are calculated

		# If the back calculated source image co-ordinates lie outside the matrix, intensity zero is assigned (Dark spot)
		if x<0 or x>xmax-1 or y<0 or y>ymax-1:
			out_arr[i,j] = 0
		else:
			# Intensities of the co-ordinates of the square that contain the back calculated source point
			I1 = in_arr[xlow,ylow] 
			I2 = in_arr[xlow,ylow+1]
			I3 = in_arr[xlow+1,ylow]
			I4 = in_arr[xlow+1,ylow+1]
			temp = Bil_interp(x,y,a,b,I1,I2,I3,I4)    # Bilinear interpolator function is invoked to obtain the target intensity
			if temp<0:
				out_arr[i,j] = 0
			elif temp>255:
				out_arr[i,j] = 255
			else:
				out_arr[i,j] = round(temp)            # Intensities are rounded to the nearest integer

plt.figure(4)
plt.imshow(out_arr,cmap='gray')                       # Target image is plotted
plt.title('Rotated pisa')

data = im.fromarray(out_arr)
data = data.convert("L")
data.save('pisa_rotated.png')                         # Target image is saved into a .png file

'''
************************************************************************************
Image scaling:
************************************************************************************
'''

image = im.open('cells_scale.png').convert("L")       # Source image opened as an 8-bit grayscale intensity matrix
in_arr = np.asarray(image)                            # Converted into numpy array         
dim = np.shape(in_arr) 
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions are obtained
xmid = (xmax-1)/2
ymid = (ymax-1)/2                                     # Image center co-ordinates are found

plt.figure(5)
plt.imshow(in_arr,cmap='gray')                        # Image matrix is plotted 
plt.title('Cells')

sx = 0.8
sy = 1.3                                              # Scaling factors

out_arr = np.zeros(dim)                               # Target image matrix initialized with zeros

# For loop to run over each co-ordinate of the target image
for i in range(xmax):
	for j in range(ymax):

		# Inverse scaling w.r.t image center (Target-to-source mapping) is performed
		x = xmid+((i-xmid)/sx)
		y = ymid+((j-ymid)/sy)

		xlow = m.floor(x)
		ylow = m.floor(y)
		a = x-xlow
		b = y-ylow                                    # a,b values are calculated

		# If the back calculated source image co-ordinates lie outside the matrix, intensity zero is assigned (Dark spot)
		if x<0 or x>xmax-1 or y<0 or y>ymax-1:
			out_arr[i,j] = 0
		else:
			# Intensities of the co-ordinates of the square that contain the back calculated source point
			I1 = in_arr[xlow,ylow] 
			I2 = in_arr[xlow,ylow+1]
			I3 = in_arr[xlow+1,ylow]
			I4 = in_arr[xlow+1,ylow+1]
			temp = Bil_interp(x,y,a,b,I1,I2,I3,I4)    # Bilinear interpolator function is invoked to obtain the target intensity
			if temp<0:
				out_arr[i,j] = 0
			elif temp>255:
				out_arr[i,j] = 255
			else:
				out_arr[i,j] = round(temp)            # Intensities are rounded to the nearest integer

plt.figure(6)
plt.imshow(out_arr,cmap='gray')                       # Target image is plotted
plt.title('Scaled cells')

plt.show()

data = im.fromarray(out_arr)
data = data.convert("L")
data.save('cells_scaled.png')                         # Target image is saved into a .png file