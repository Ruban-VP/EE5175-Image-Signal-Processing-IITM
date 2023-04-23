'''
************************************************************************************
EE5175: Image signal processing - Lab 08 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 22/04/2023
Execution command: python3 EE19B138_Lab_08_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import cv2
import numpy as np
import scipy.linalg as sp
import scipy.io as spio
import random as rand
from PIL import Image as im
from moviepy.editor import ImageSequenceClip
import math as m
import matplotlib.pyplot as plt      
from sift import sift as sift_corresp                 # Importing the required libraries

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
Image warper function:
************************************************************************************
'''

def IWF(K,n,d,tr_vec):
	
	t = np.transpose(np.array([tr_vec[0:3]]))

	rx = tr_vec[3]
	ry = tr_vec[4]
	rz = tr_vec[5]                                    # Rotation angles obtained

	# Rotation matrices in different axes formed
	Rx = np.array([[1,0,0],[0,m.cos(rx),-m.sin(rx)],[0,m.sin(rx),m.cos(rx)]])
	Ry = np.array([[m.cos(ry),0,m.sin(ry)],[0,1,0],[-m.sin(ry),0,m.cos(ry)]])
	Rz = np.array([[m.cos(rz),-m.sin(rz),0],[m.sin(rz),m.cos(rz),0],[0,0,1]])
	R = np.matmul(np.matmul(Rz,Ry),Rx)

	# Homography computed based on the formula given
	M = np.linalg.inv(R + (np.matmul(t,n)/d))
	H = np.matmul(np.matmul(K,M),np.linalg.inv(K))

	return H

'''
************************************************************************************
RANSAC algorithm function:
************************************************************************************
'''

def RANSAC(corresp_1,corresp_2,eps,frac,N_iter):
	dim = np.shape(corresp_1)
	num = dim[0]                                         # Number of point correspondences
	iter_count = 0

	flag = 0
	while flag==0 and iter_count<=N_iter:                # While loop to compute homography matrix using RANSAC
		
		iter_count = iter_count+1
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
Simulated motion blur:
************************************************************************************
'''

image = im.open('pillars.jpg').convert("L")           # Source image opened as an 8-bit grayscale intensity matrix
inp = np.asarray(image)                               # Converted into numpy array
dim = np.shape(inp)
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions obtained
xmid = xmax/2
ymid = ymax/2

with open('cam.txt') as f:
	cam_data = f.readlines()                          # Camera motion parameters obtained

num = int(len(cam_data)/6)-1
FPS = 20

tx_data = np.array([float(x) for x in cam_data[1:num+1]])
ty_data = np.array([float(x) for x in cam_data[num+2:2*num+2]])
tz_data = np.array([float(x) for x in cam_data[2*num+3:3*num+3]])
rx_data = np.array([float(x) for x in cam_data[3*num+4:4*num+4]])
ry_data = np.array([float(x) for x in cam_data[4*num+5:5*num+5]])
rz_data = np.array([float(x) for x in cam_data[5*num+6:6*num+6]])

# Camera motion parameter array formed
cam_data = np.array([tx_data,ty_data,tz_data,rx_data,ry_data,rz_data])

d = 1000
n = np.array([[0,0,1]])
f = 500
K = np.array([[f,0,0],[0,f,0],[0,0,1]])               # Camera intrinsics specified

out_imgs = np.zeros([num,xmax,ymax,1])

for ind in range(num):

	tr_vec = cam_data[:,ind]
	H = IWF(K,n,d,tr_vec)                             # Homography computed for current parameters

	for i in range(xmax):
		for j in range(ymax):

			i_cent = i-xmid
			j_cent = j-ymid
			targ_array = np.array([[i_cent],[j_cent],[1]]) # Target co-ordinate vector w.r.t image center are formed

			i_array = np.dot(H,targ_array)

			[x,y] = i_array[:2,0]/i_array[2,0]         # Source co-ordinate vector obtained
			x = x+xmid                                 # Source co-ordinates w.r.t the matrix origin, i.e., 
			y = y+ymid                                 # top left corner are computed
												 

			# The source co-ordinates of first image are used to get the intensity from it
			xlow = m.floor(x)
			ylow = m.floor(y)
			if xlow==xmax-1: 
				xlow = xmax-2
			if ylow==ymax-1:
				ylow = ymax-2
			a = x-xlow
			b = y-ylow 

			# Intensities of the co-ordinates of the square that contain the back calculated source point
			if not (x<0 or x>xmax-1 or y<0 or y>ymax-1):
				I1 = inp[xlow,ylow] 
				I2 = inp[xlow,ylow+1]
				I3 = inp[xlow+1,ylow]
				I4 = inp[xlow+1,ylow+1]
				
				out_imgs[ind,i,j,0] = Bil_interp(a,b,I1,I2,I3,I4)

out1 = np.mean(out_imgs, axis=0)
out1 = out1[:,:,0]                                    # Motion blurred image obtained

clip = ImageSequenceClip(list(out_imgs), fps=20)
clip.write_gif('Blur.gif', fps=FPS)                   # Saved as .gif file

# Occurences of distinct poses obtained
cam_data_un, cam_data_counts = np.unique(cam_data, return_counts=True, axis=1)
un_num = np.shape(cam_data_un)[1]
out_imgs = np.zeros([un_num,xmax,ymax,1])

for ind in range(un_num):

	tr_vec = cam_data_un[:,ind]
	H = IWF(K,n,d,tr_vec)                             # Homography computed for current parameters

	for i in range(xmax):
		for j in range(ymax):

			i_cent = i-xmid
			j_cent = j-ymid
			targ_array = np.array([[i_cent],[j_cent],[1]]) # Target co-ordinate vector w.r.t image center are formed

			i_array = np.dot(H,targ_array)
			
			[x,y] = i_array[:2,0]/i_array[2,0]         # Source co-ordinate vector obtained
			x = x+xmid                                 # Source co-ordinates w.r.t the matrix origin, i.e., 
			y = y+ymid                                 # top left corner are computed                                    

			# The source co-ordinates of first image are used to get the intensity from it
			xlow = m.floor(x)
			ylow = m.floor(y)
			if xlow==xmax-1: 
				xlow = xmax-2
			if ylow==ymax-1:
				ylow = ymax-2
			a = x-xlow
			b = y-ylow 

			# Intensities of the co-ordinates of the square that contain the back calculated source point
			if not (x<0 or x>xmax-1 or y<0 or y>ymax-1):
				I1 = inp[xlow,ylow] 
				I2 = inp[xlow,ylow+1]
				I3 = inp[xlow+1,ylow]
				I4 = inp[xlow+1,ylow+1]
				
				out_imgs[ind,i,j,0] = cam_data_counts[ind]*Bil_interp(a,b,I1,I2,I3,I4)

out2 = np.sum(out_imgs, axis=0)/num
out2 = out2[:,:,0]                                    # Motion blurred image obtained

err = np.linalg.norm(out1-out2)**2
err = err/(xmax*ymax)
print(err)                                            # MSE obtained and printed

# Camera motion with only depth changes and in-plane translations
out_imgs = np.zeros([num,xmax,ymax,1,3])
cam_data_temp = cam_data
cam_data_temp[2:6,:] = 0

d_vals = [200,1000,5000]
n = np.array([[0,0,1]])
f = 500
K = np.array([[f,0,0],[0,f,0],[0,0,1]])

for d_ind in range(3):

	d = d_vals[d_ind]
	for ind in range(num):

		tr_vec = cam_data_temp[:,ind]
		H = IWF(K,n,d,tr_vec)                         # Homography computed for current parameters

		for i in range(xmax):
			for j in range(ymax):

				i_cent = i-xmid
				j_cent = j-ymid
				targ_array = np.array([[i_cent],[j_cent],[1]]) # Target co-ordinate vector w.r.t image center are formed

				i_array = np.dot(H,targ_array)
				
				[x,y] = i_array[:2,0]/i_array[2,0]    # Source co-ordinate vector obtained
				x = x+xmid                            # Source co-ordinates w.r.t the matrix origin, i.e., 
				y = y+ymid                            # top left corner are computed                                      

				# The source co-ordinates of first image are used to get the intensity from it
				xlow = m.floor(x)
				ylow = m.floor(y)
				if xlow==xmax-1: 
					xlow = xmax-2
				if ylow==ymax-1:
					ylow = ymax-2
				a = x-xlow
				b = y-ylow 

				# Intensities of the co-ordinates of the square that contain the back calculated source point
				if not (x<0 or x>xmax-1 or y<0 or y>ymax-1):
					I1 = inp[xlow,ylow] 
					I2 = inp[xlow,ylow+1]
					I3 = inp[xlow+1,ylow]
					I4 = inp[xlow+1,ylow+1]
					
					out_imgs[ind,i,j,0,d_ind] = Bil_interp(a,b,I1,I2,I3,I4)

out3 = np.mean(out_imgs, axis=0)
out3_1 = out3[:,:,0,0]
out3_2 = out3[:,:,0,1]
out3_3 = out3[:,:,0,2]                                # Motion blurred images obtained

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,0]), fps=20)
clip.write_gif('Depth_1.gif', fps=FPS)

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,1]), fps=20)
clip.write_gif('Depth_2.gif', fps=FPS)

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,2]), fps=20)
clip.write_gif('Depth_3.gif', fps=FPS)                # Saved as .gif files

# Camera motion with only focal length changes and in-plane rotations
out_imgs = np.zeros([num,xmax,ymax,1,3])
cam_data_temp = cam_data
cam_data_temp[0:4,:] = 0
cam_data_temp[5,:] = 0

d = 1000
n = np.array([[0,0,1]])
f_vals = [100,500,1000]
K = np.array([[f,0,0],[0,f,0],[0,0,1]])

for f_ind in range(3):

	f = f_vals[f_ind]
	for ind in range(num):

		tr_vec = cam_data_temp[:,ind]
		H = IWF(K,n,d,tr_vec)                         # Homography computed for current parameters

		for i in range(xmax):
			for j in range(ymax):

				i_cent = i-xmid
				j_cent = j-ymid 
				targ_array = np.array([[i_cent],[j_cent],[1]]) # Target co-ordinate vector w.r.t image center are formed

				i_array = np.dot(H,targ_array)

				[x,y] = i_array[:2,0]/i_array[2,0]    # Source co-ordinate vector obtained
				x = x+xmid                            # Source co-ordinates w.r.t the matrix origin, i.e., 
				y = y+ymid                            # top left corner are computed                                     

				# The source co-ordinates of first image are used to get the intensity from it
				xlow = m.floor(x)
				ylow = m.floor(y)
				if xlow==xmax-1: 
					xlow = xmax-2
				if ylow==ymax-1:
					ylow = ymax-2
				a = x-xlow
				b = y-ylow 

				# Intensities of the co-ordinates of the square that contain the back calculated source point
				if not (x<0 or x>xmax-1 or y<0 or y>ymax-1):
					I1 = inp[xlow,ylow] 
					I2 = inp[xlow,ylow+1]
					I3 = inp[xlow+1,ylow]
					I4 = inp[xlow+1,ylow+1]
					
					out_imgs[ind,i,j,0,f_ind] = Bil_interp(a,b,I1,I2,I3,I4)

out4 = np.mean(out_imgs, axis=0)
out4_1 = out4[:,:,0,0]
out4_2 = out4[:,:,0,1]
out4_3 = out4[:,:,0,2]                                # Motion blurred images obtained

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,0]), fps=20)
clip.write_gif('Focus_1.gif', fps=FPS)

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,1]), fps=20)
clip.write_gif('Focus_2.gif', fps=FPS)

clip = ImageSequenceClip(list(out_imgs[:,:,:,:,2]), fps=20)
clip.write_gif('Focus_3.gif', fps=FPS)                # Saved as .gif files

## Motion blurred images saved as .png files
data = im.fromarray(out1)
data = data.convert("L")
data.save('Blur_1.png')                                  

data = im.fromarray(out2)
data = data.convert("L")
data.save('Blur_2.png') 

data = im.fromarray(out3_1)
data = data.convert("L")
data.save('Depth_1.png') 

data = im.fromarray(out3_2)
data = data.convert("L")
data.save('Depth_2.png') 

data = im.fromarray(out3_3)
data = data.convert("L")
data.save('Depth_3.png')  

data = im.fromarray(out4_1)
data = data.convert("L")
data.save('Focus_1.png') 

data = im.fromarray(out4_2)
data = data.convert("L")
data.save('Focus_2.png') 

data = im.fromarray(out4_3)
data = data.convert("L")
data.save('Focus_3.png')   

'''
************************************************************************************
Realistic motion blur:
************************************************************************************
'''	  

data = spio.loadmat("frames.mat")                     # frames.mat file is unzipped
dim = np.shape(data["frame01"])
xmax = dim[0]
ymax = dim[1]                                         # Image dimensions obtained
xmid = xmax/2
ymid = ymax/2                                         # Image centers obtained                                        

frames = np.zeros([xmax,ymax,3,11], dtype=np.uint8)
for i in range(1,12):
	if i>=1 and i<=9:
		frames[:,:,:,i-1] = data["frame0"+str(i)]
	else:
		frames[:,:,:,i-1] = data["frame"+str(i)]      # Frames obtained

mod_frames = np.zeros([xmax,ymax,3,11], dtype=np.uint8)

eps = 10
frac = 0.8
N_iter = 50
inp = frames[:,:,:,5]
curr_frame = frames[:,:,:,4]

for ind in range(11):

	curr_frame = frames[:,:,:,ind]
	[corresp_1, corresp_2] = sift_corresp(curr_frame,inp)  # Point-to-point correspondences obtained

	corresp_1[:,0] = corresp_1[:,0]-xmid
	corresp_1[:,1] = corresp_1[:,1]-ymid

	corresp_2[:,0] = corresp_2[:,0]-xmid
	corresp_2[:,1] = corresp_2[:,1]-ymid

	H = RANSAC(corresp_2,corresp_1,eps,frac,N_iter)   # Homography obtained using RANSAC

	for i in range(xmax):
		for j in range(ymax):

			i_cent = i-xmid
			j_cent = j-ymid
			targ_array = np.array([[i_cent],[j_cent],[1]]) # Target co-ordinate vector w.r.t image center are formed

			i_array = np.dot(H,targ_array)

			[x,y] = i_array[:2,0]/i_array[2,0]        # Source co-ordinate vector obtained
			x = x+xmid                                # Source co-ordinates w.r.t the matrix origin, i.e., 
			y = y+ymid                                # top left corner are computed                                   

			# The source co-ordinates of first image are used to get the intensity from it
			xlow = m.floor(x)
			ylow = m.floor(y)
			if xlow==xmax-1: 
				xlow = xmax-2
			if ylow==ymax-1:
				ylow = ymax-2
			a = x-xlow
			b = y-ylow 

			# Intensities of the co-ordinates of the square that contain the back calculated source point
			if not (x<0 or x>xmax-1 or y<0 or y>ymax-1):
				I1 = inp[xlow,ylow,:] 
				I2 = inp[xlow,ylow+1,:]
				I3 = inp[xlow+1,ylow,:]
				I4 = inp[xlow+1,ylow+1,:]
				
				mod_frames[i,j,:,ind] = Bil_interp(a,b,I1,I2,I3,I4)

err_stack = frames-mod_frames
err_stack = np.linalg.norm(err_stack, axis=0)**2
err_stack = np.sum(err_stack, axis=0)
err_stack = np.sum(err_stack, axis=0)/(xmax*ymax*3)   # MSE between the blurred images computed

frame_inds = [i for i in range(1,12)]
plt.figure(1)
plt.plot(frame_inds, err_stack)
plt.xlabel("Frame index")
plt.ylabel("Mean squared error")
plt.title("Frame vs MSE plot")                        # MSE between different frames plotted agaisnt frame index

plt.figure(2)
plt.imshow(frames[:,:,:,0])

plt.figure(3)
plt.imshow(mod_frames[:,:,:,0])

plt.figure(4)
plt.imshow(frames[:,:,:,5])

plt.figure(5)
plt.imshow(mod_frames[:,:,:,5])

plt.figure(6)
plt.imshow(frames[:,:,:,10])

plt.figure(7)
plt.imshow(mod_frames[:,:,:,10])                      # Different sets of similar frames plotted

plt.show()

direct = np.mean(frames, axis=3)
direct = direct.astype(np.uint8)                      # Direct average of stack

indirect = np.mean(mod_frames, axis=3)
indirect = indirect.astype(np.uint8)                  # Indirect average of stack based on homography

direct = np.flip(direct, axis=2)
indirect = np.flip(indirect, axis=2)

cv2.imwrite("direct_blur.png", direct)
cv2.imwrite("indirect_blur.png", indirect)            # Saved as .png files