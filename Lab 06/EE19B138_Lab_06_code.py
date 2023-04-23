'''
************************************************************************************
EE5175: Image signal processing - Lab 06 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 08/03/2023
Execution command: python3 EE19B138_Lab_06_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
import scipy.io as sp
from PIL import Image as im
import math as m
import matplotlib.pyplot as plt                       # Importing the required libraries

'''
************************************************************************************
Sum-modified Laplacian operator function:
************************************************************************************
'''

def SML(inp):
	dim = np.shape(inp)
	xmax = dim[0]
	ymax = dim[1]                                     # Input image dimensions are obtained

	# Input image zero padded for computing Laplacain at image edges
	inp = np.concatenate((np.zeros([1,ymax]),inp,np.zeros([1,ymax])),axis=0) 
	inp = np.concatenate((np.zeros([xmax+2,1]),inp,np.zeros([xmax+2,1])),axis=1)

	lap_out = np.zeros([xmax,ymax])

	# For loop to compute SML value for each pixel
	for i in range(xmax):
		for j in range(ymax):
			val1 = np.absolute(inp[i,j+1]+inp[i+2,j+1]-2*inp[i+1,j+1])
			val2 = np.absolute(inp[i+1,j]+inp[i+1,j+2]-2*inp[i+1,j+1])   
			lap_out[i,j] = val1+val2

	return lap_out

'''
************************************************************************************
Gaussian function based depth estimator function:
************************************************************************************
'''

def optimal_d(fm,fm_prev,fm_next,dm,del_d):
	dm_prev = dm-del_d
	dm_next = dm+del_d
	A = np.log(fm/fm_prev)
	B = np.log(fm/fm_next)

	d_opt = (dm/2)+((A*dm_next+B*dm_prev)/(2*(A+B)))  # Gaussian function based depth estimate calculation
	return d_opt

'''
************************************************************************************
Shape from focus:
************************************************************************************
'''

data = sp.loadmat("stack.mat")                        # stack.mat file is unzipped
num = data["numframes"][0,0]                          # Number of frames obtained
dim = np.shape(data["frame001"])
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions obtained

del_d = 50.50                                         # Depth step-size

mat_q0 = np.zeros([num,xmax,ymax])
mat_q1 = np.zeros([num,xmax,ymax])
mat_q2 = np.zeros([num,xmax,ymax])                    # 3-D arrays to store SML values for every frame

# For loop to find SML focus measure for each frame
for i in range(num):
	if i<9:
		frame_num = str("frame")+str(0)+str(0)+str(i+1)
	elif i>=9 and i<99:
		frame_num = str("frame")+str(0)+str(i+1)
	else:
		frame_num = str("frame")+str(i+1)

	curr_img = data[frame_num]
	SML_out = SML(curr_img)

	mat_q0[i,:,:] = SML_out                           # For q=0, SML profile is the focus measure

	# SML output zero padded to compute window sum
	SML_out = np.concatenate((np.zeros([2,ymax]),SML_out,np.zeros([2,ymax])),axis=0) 
	SML_out = np.concatenate((np.zeros([xmax+4,2]),SML_out,np.zeros([xmax+4,2])),axis=1)

	# For loop to find the window sum for each pixel on the image
	for x in range(xmax):
		for y in range(ymax):
			mat_q1[i,x,y] = np.sum(SML_out[x+1:x+4,y+1:y+4])
			mat_q2[i,x,y] = np.sum(SML_out[x:x+5,y:y+5])

depth_map_q0 = np.zeros([xmax,ymax])
depth_map_q1 = np.zeros([xmax,ymax])
depth_map_q2 = np.zeros([xmax,ymax])                  # Depth map arrays initialized for different q values

# For loop to compute depth estimate for each pixel 
for x in range(xmax):
	for y in range(ymax):
		dm_q0 = np.argmax(mat_q0[:,x,y])
		dm_q1 = np.argmax(mat_q1[:,x,y])
		dm_q2 = np.argmax(mat_q2[:,x,y])              # Depth index at which the focus measure is maximum is obtained

		# If max. focus depth index is starting or ending frame, it is taken as the optimal depth estimate
		if dm_q0==0 or dm_q0==99:
			depth_map_q0[x,y] = dm_q0
		# Else optimal_d function is used to find the depth estimate
		else:
			depth_map_q0[x,y] = optimal_d(mat_q0[dm_q0,x,y],mat_q0[dm_q0-1,x,y],mat_q0[dm_q0+1,x,y],del_d*dm_q0,del_d)
		
		# If max. focus depth index is starting or ending frame, it is taken as the optimal depth estimate
		if dm_q1==0 or dm_q1==99:
			depth_map_q1[x,y] = dm_q1
		# Else optimal_d function is used to find the depth estimate
		else:
			depth_map_q1[x,y] = optimal_d(mat_q1[dm_q1,x,y],mat_q1[dm_q1-1,x,y],mat_q1[dm_q1+1,x,y],del_d*dm_q1,del_d)

		# If max. focus depth index is starting or ending frame, it is taken as the optimal depth estimate
		if dm_q2==0 or dm_q2==99:
			depth_map_q2[x,y] = dm_q2
		# Else optimal_d function is used to find the depth estimate
		else:
			depth_map_q2[x,y] = optimal_d(mat_q2[dm_q2,x,y],mat_q2[dm_q2-1,x,y],mat_q2[dm_q2+1,x,y],del_d*dm_q2,del_d)

xline = range(xmax)
yline = range(ymax)
X,Y = np.meshgrid(xline,yline) 
c_map = "plasma"

# Code to plot the 3-D depth maps for different q values is present below

fig = plt.figure(1)
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(X, Y, np.transpose(depth_map_q0), cmap=plt.cm.get_cmap(c_map))

axes.set_xlabel(r'$X\rightarrow$')
axes.set_ylabel(r'$Y\rightarrow$')
axes.set_zlabel(r'$Z\rightarrow$')
axes.set_title("Depth map for q=0")
plt.savefig("Depth_map_q0.png")

fig = plt.figure(2)
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(X, Y, np.transpose(depth_map_q1), cmap=plt.cm.get_cmap(c_map))

axes.set_xlabel(r'$X\rightarrow$')
axes.set_ylabel(r'$Y\rightarrow$')
axes.set_zlabel(r'$Z\rightarrow$')
axes.set_title('Depth map for q=1')
plt.savefig("Depth_map_q1.png")

fig = plt.figure(3)
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(X, Y, np.transpose(depth_map_q2), cmap=plt.cm.get_cmap(c_map))

axes.set_xlabel(r'$X\rightarrow$')
axes.set_ylabel(r'$Y\rightarrow$')
axes.set_zlabel(r'$Z\rightarrow$')
axes.set_title('Depth map for q=2')                
plt.savefig("Depth_map_q2.png")

plt.show()