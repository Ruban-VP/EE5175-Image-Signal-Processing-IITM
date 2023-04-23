'''
************************************************************************************
EE5175: Image signal processing - Lab 02 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 10/02/2023
Execution command: python3 EE19B138_Lab_02_code.py
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

def Bil_interp(a,b,I1,I2,I3,I4):
	out_int = ((1-a)*(1-b)*I1)+((1-a)*b*I2)+(a*(1-b)*I3)+(a*b*I4)  # Bilinear interpolation formula
	return out_int

'''
************************************************************************************
Occlusion detection:
************************************************************************************
'''

image = im.open('IMG1.png').convert("L")                 # First image opened as an 8-bit grayscale intensity matrix
img1= np.asarray(image)                                  # Converted into numpy array
dim1 = np.shape(img1)
xmax1 = dim1[0]
ymax1 = dim1[1]                                          # Image dimensions are obtained

image = im.open('IMG2.png').convert("L")                 # Second image opened as an 8-bit grayscale intensity matrix
img2 = np.asarray(image)                                 # Converted into numpy array
dim2 = np.shape(img2)
xmax2 = dim2[0]
ymax2 = dim2[1]                                          # Image dimensions are obtained

pos1 = np.array([[29,124],[157,372]], dtype='float')     # Points given on image 1
pos2 = np.array([[93,248],[328,399]], dtype='float')     # Points given on image 2 

# The matrix used for parameters (theta, tx and ty) estimation is formed 
H = np.array([[pos1[0,0],pos1[0,1],1,0],[pos1[0,1],-pos1[0,0],0,1],[pos1[1,0],pos1[1,1],1,0],[pos1[1,1],-pos1[1,0],0,1]], dtype='float')
# Output vector of this estimation equation is formed
v = np.array([[pos2[0,0]],[pos2[0,1]],[pos2[1,0]],[pos2[1,1]]], dtype='float')

params = np.dot(np.linalg.inv(H),v)                      # Parameters are obtained as a 4x1 vector              

if params[0,0]>0:
	theta = m.asin(params[1,0])
else:
	theta = m.pi-m.asin(params[1,0])                     # Rotation angle is found (in Radians)
theta_deg = theta*180/m.pi                               # Rotation angle in degrees
tx = round(params[2,0])                                  # Translation along x is found (in pixels)
ty = round(params[3,0])                                  # Translation along y is found (in pixels) 

corr_img = np.zeros([xmax1,ymax1]);                      # Corrected image array initialized with zeros

# Homogenous matrix that describes the similarity transformation between image 1 and image 2
Hom_matrix = np.array([[m.cos(theta),m.sin(theta),tx],[-m.sin(theta),m.cos(theta),ty],[0,0,1]], dtype='float')

# For loop to run over each co-ordinate of the target image
for i in range(xmax1):
	for j in range(ymax1):

		pos_vec = np.array([[i],[j],[1]],dtype='float')
		trans_pos_vec = np.dot(Hom_matrix,pos_vec)       # Target-to-source mapping

		x = trans_pos_vec[0,0]
		y = trans_pos_vec[1,0]                           # Source co-ordinates are found

		xlow = m.floor(x)
		ylow = m.floor(y)
		a = x-xlow
		b = y-ylow                                       # a,b values are calculated

		# If the back calculated source image co-ordinates lie outside the source image, intensity zero is assigned (Dark spot)
		if x<0 or x>xmax2-1 or y<0 or y>ymax2-1:
			corr_img[i,j] = 0
		else:
			# Intensities of the co-ordinates of the square that contain the back calculated source point
			I1 = img2[xlow,ylow] 
			I2 = img2[xlow,ylow+1]
			I3 = img2[xlow+1,ylow]
			I4 = img2[xlow+1,ylow+1]

			# Bilinear interpolator function is invoked to obtain the target intensity
			temp = Bil_interp(a,b,I1,I2,I3,I4)           
			corr_img[i,j] = round(temp)                  # Intensities are rounded to the nearest integer

Change_matrix = np.abs(img1-corr_img)                    # Changes in image 2 w.r.t image 1

plt.figure(1)
plt.imshow(img1,cmap='gray')                             # First image is plotted
plt.title('First image')                                 

plt.figure(2)
plt.imshow(img2,cmap='gray')                             # Second image is plotted
plt.title('Second image')

plt.figure(3)
plt.imshow(corr_img,cmap='gray')                         # Corrected second image is plotted
plt.title('Second image corrected')

plt.figure(4)
plt.imshow(Change_matrix,cmap='gray')                    # Image change matrix is plotted
plt.title("Changes between the images")                 

plt.show()

data = im.fromarray(corr_img)
data = data.convert("L")
data.save('Corrected_IMG2.png')                          # Corrected second image is saved into a .png file

data = im.fromarray(Change_matrix)
data = data.convert("L")
data.save('Changes.png')                                 # Change image is saved into a .png file

print("Rotation angle (in degrees): "+str(theta_deg))
print("Translation along x        : "+str(tx))
print("Translation along y        : "+str(ty))           # Parameter values are printed on screen