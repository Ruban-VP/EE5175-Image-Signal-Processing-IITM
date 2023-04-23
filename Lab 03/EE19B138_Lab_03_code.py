'''
************************************************************************************
EE5175: Image signal processing - Lab 03 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 21/02/2023
Execution command: python3 EE19B138_Lab_03_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
import scipy.linalg as sp
import random as rand
from PIL import Image as im
import math as m
import matplotlib.pyplot as plt      
from sift import sift as sift_corresp # Importing the required libraries

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
RANSAC algorithm function:
************************************************************************************
'''

def RANSAC(corresp_1,corresp_2,eps,frac):
    dim = np.shape(corresp_1)
    num = dim[0]                                         # Number of point correspondences

    flag = 0
    while flag==0:                                       # While loop to compute homography matrix using RANSAC
        count = 0
        randints = rand.sample(range(num),4)             # Random distinct four points are chosen
        points1 = corresp_1[randints,:]
        points2 = corresp_2[randints,:]
        
        A = np.zeros([8,9])                               
        for i in range(4):
            A[i*2,0] = -points1[i,0]
            A[i*2,1] = -points1[i,1]
            A[i*2,2] = -1
            A[i*2,6] = points1[i,0]*points2[i,0]
            A[i*2,7] = points1[i,1]*points2[i,0]
            A[i*2,8] = points2[i,0]

            A[i*2+1,3] = -points1[i,0]
            A[i*2+1,4] = -points1[i,1]
            A[i*2+1,5] = -1
            A[i*2+1,6] = points1[i,0]*points2[i,1]
            A[i*2+1,7] = points1[i,1]*points2[i,1]
            A[i*2+1,8] = points2[i,1]                    # 'A' Matrix is formed using the point correspondences

        h = sp.null_space(A)
        h = h[:,0]                                       # Nullspace of the 'A' matrix is found

        # The homography matrix is formed from the nullspace 'h' vector
        H = np.array([[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],h[8]]])  

        # Remaining point correspondences are used to form the consensus set
        inds = [i for i in range(num) if i not in randints]  
        
        for i in inds:
            act_x2 = np.array([[corresp_2[i,0]],[corresp_2[i,1]]])       # Actual co-ordinates
            x2 = np.array([[corresp_1[i,0]],[corresp_1[i,1]],[1]])
            obt_x2 = np.dot(H,x2)
            obt_x2 = np.array([[obt_x2[0,0]],[obt_x2[1,0]]])/obt_x2[2,0] # Calculated co-ordinates
            error = np.linalg.norm(act_x2-obt_x2)                        # Magnitude of the error

            if error<eps:
                count = count+1                          # If error is less than epsilon, it is added to consensus

        calc_frac = count/(num-4)                        # Consensus set fraction
        if calc_frac>frac:
            flag = 1       # If consensus fraction is more than required, flag is made 1, i.e., exit the while loop

    return H

'''
************************************************************************************
Image reading and Homography computation:
************************************************************************************
'''

image = im.open('img1.png').convert("L")                 # First image opened as an 8-bit grayscale intensity matrix
img1= np.asarray(image)                                  # Converted into numpy array

image = im.open('img2.png').convert("L")                 # Second image opened as an 8-bit grayscale intensity matrix
img2 = np.asarray(image)                                 # Converted into numpy array

image = im.open('img3.png').convert("L")                 # Third image opened as an 8-bit grayscale intensity matrix
img3 = np.asarray(image)                                 # Converted into numpy array

dim = np.shape(img1)
xmax = dim[0]
ymax = dim[1]                                            # Image dimensions are obtained
xmid = xmax/2
ymid = ymax/2                                            # Center co-ordinates are computed

[corresp_12_1, corresp_12_2] = sift_corresp(img1,img2)
[corresp_23_2, corresp_23_3] = sift_corresp(img2,img3)   # Obtaining correspondences using SIFT

corresp_12_1[:,0] = corresp_12_1[:,0]-xmid
corresp_12_1[:,1] = corresp_12_1[:,1]-ymid

corresp_12_2[:,0] = corresp_12_2[:,0]-xmid
corresp_12_2[:,1] = corresp_12_2[:,1]-ymid

corresp_23_2[:,0] = corresp_23_2[:,0]-xmid
corresp_23_2[:,1] = corresp_23_2[:,1]-ymid

corresp_23_3[:,0] = corresp_23_3[:,0]-xmid
corresp_23_3[:,1] = corresp_23_3[:,1]-ymid               # Co-ordinates found w.r.t image center

eps = 5
frac = 0.9
H21 = RANSAC(corresp_12_2,corresp_12_1,eps,frac)
H23 = RANSAC(corresp_23_2,corresp_23_3,eps,frac)         # RANSAC algorithm is used to compute the homographies

'''
************************************************************************************
Image mosaicing:
************************************************************************************
'''

xmax_out = m.ceil(1.6*xmax)
ymax_out = m.ceil(2.15*ymax)                             # Final image dimensions are set
xmid_out = xmax_out/2   
ymid_out = ymax_out/2                                    # Image center co-ordinates are found

out_img = np.zeros([xmax_out,ymax_out])                  # Empty canvas is created
offset_x = 10
offset_y = 90                                            # Offset values are set

for ind_i in range(xmax_out):
    for ind_j in range(ymax_out):

        i = ind_i + offset_x
        j = ind_j + offset_y                             # Offset is applied for adjustments 

        i_cent = i-xmid_out
        j_cent = j-ymid_out
        targ_array = np.array([[i_cent],[j_cent],[1]])   # Target co-ordinate vector w.r.t image center are formed

        i1_array = np.dot(H21,targ_array)
        i2_array = targ_array                            # The source co-ordinate vectors on different images are 
        i3_array = np.dot(H23,targ_array)                # obtained using the homography matrices 

        [x1,y1] = i1_array[:2,0]/i1_array[2,0]
        [x2,y2] = i2_array[:2,0]/i2_array[2,0]
        [x3,y3] = i3_array[:2,0]/i3_array[2,0]           # Scale factor is removed to get image co-ordinates   

        x1 = x1+xmid
        x2 = x2+xmid
        x3 = x3+xmid
        y1 = y1+ymid
        y2 = y2+ymid                                     # Source co-ordinates w.r.t the matrix origin, i.e., 
        y3 = y3+ymid                                     # top left corner are computed

        # Count: Variable to keep track of the number of images in which the target co-ordinates are present when 
        # back calculated using Target-to-source mapping
        count = 0                                        

    # The source co-ordinates of first image are used to get the intensity from it
        xlow1 = m.floor(x1)
        ylow1 = m.floor(y1)
        if xlow1==xmax-1: 
            xlow1 = xmax-2
        if ylow1==ymax-1:
            ylow1 = ymax-2
        a1 = x1-xlow1
        b1 = y1-ylow1 

        # If the back calculated source image co-ordinates lie outside the source image, intensity zero is assigned (Dark spot)  
        if x1<0 or x1>xmax-1 or y1<0 or y1>ymax-1:
            temp1 = 0
        else:  
            # Intensities of the co-ordinates of the square that contain the back calculated source point
            I1 = img1[xlow1,ylow1] 
            I2 = img1[xlow1,ylow1+1]
            I3 = img1[xlow1+1,ylow1]
            I4 = img1[xlow1+1,ylow1+1]

            # Bilinear interpolator function is invoked to obtain the target intensity
            temp1 = Bil_interp(a1,b1,I1,I2,I3,I4)           
            count = count+1                               
    
    # The source co-ordinates of second image are used to get the intensity from it
        xlow2 = m.floor(x2)
        ylow2 = m.floor(y2)
        if xlow2==xmax-1: 
            xlow2 = xmax-2
        if ylow2==ymax-1:
            ylow2 = ymax-2
        a2 = x2-xlow2
        b2 = y2-ylow2  

        # If the back calculated source image co-ordinates lie outside the source image, intensity zero is assigned (Dark spot)  
        if x2<0 or x2>xmax-1 or y2<0 or y2>ymax-1:
            temp2 = 0
        else:  
            # Intensities of the co-ordinates of the square that contain the back calculated source point
            I1 = img2[xlow2,ylow2] 
            I2 = img2[xlow2,ylow2+1]
            I3 = img2[xlow2+1,ylow2]
            I4 = img2[xlow2+1,ylow2+1]

            # Bilinear interpolator function is invoked to obtain the target intensity
            temp2 = Bil_interp(a2,b2,I1,I2,I3,I4)           
            count = count+1

    # The source co-ordinates of third image are used to get the intensity from it
        xlow3 = m.floor(x3)
        ylow3 = m.floor(y3)
        if xlow3==xmax-1: 
            xlow3 = xmax-2
        if ylow3==ymax-1:
            ylow3 = ymax-2
        a3 = x3-xlow3
        b3 = y3-ylow3

        # If the back calculated source image co-ordinates lie outside the source image, intensity zero is assigned (Dark spot)  
        if x3<0 or x3>xmax-1 or y3<0 or y3>ymax-1:
            temp3 = 0
        else:  
            # Intensities of the co-ordinates of the square that contain the back calculated source point
            I1 = img3[xlow3,ylow3] 
            I2 = img3[xlow3,ylow3+1]
            I3 = img3[xlow3+1,ylow3]
            I4 = img3[xlow3+1,ylow3+1]

            # Bilinear interpolator function is invoked to obtain the target intensity
            temp3 = Bil_interp(a3,b3,I1,I2,I3,I4)           
            count = count+1

    # If atleast one image contains the target co-ordinates, target intensity is assigned accordingly 
    # by blending the source intensities (Averaging method)
        if count!=0:
            out_img[ind_i,ind_j] = round((temp1+temp2+temp3)/count)

data = im.fromarray(out_img)
data = data.convert("L")
data.save('Mosaic.png')                                  # Mosaic image is saved into a .png file

plt.figure(1)
plt.imshow(img1,cmap='gray')                             # First image is plotted

plt.figure(2)
plt.imshow(img2,cmap='gray')                             # Second image is plotted

plt.figure(3)
plt.imshow(img3,cmap='gray')                             # Third image is plotted

plt.figure(4)
plt.imshow(out_img,cmap='gray')                          # Mosaic image is plotted

plt.show()