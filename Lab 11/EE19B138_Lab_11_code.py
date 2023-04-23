'''
************************************************************************************
EE5175: Image signal processing - Lab 11 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 09/04/2023
Execution command: python3 EE19B138_Lab_11_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import cv2
import numpy as np
from PIL import Image as im
import math as m
import random as rand
import matplotlib.pyplot as plt                       # Importing the required libraries

'''
************************************************************************************
K-means clustering function:
************************************************************************************
'''

def K_means(img,xmax,ymax,K,ini_vecs,num):

	cent_vecs = ini_vecs
	cent_vals = np.zeros([xmax,ymax])
	diff_arr = np.zeros([xmax,ymax,3,K])
	out_img = np.zeros([xmax,ymax,3], dtype = np.uint8)

	for iter_num in range(num):

		for k in range(K):
			temp = np.array([[cent_vecs[:,k]]])
			diff_arr[:,:,:,k] = img-temp

		diff_arr_norm = np.linalg.norm(diff_arr, axis=2)       # Distance measures computed
		cent_vals = np.argmin(diff_arr_norm, axis=2)           # Cluster indices assigned

		for k in range(K):
			locs = np.where(cent_vals==k)
			count = np.count_nonzero(cent_vals==k)
			temp = np.sum(img[locs], axis=0)
			if count!=0:
				cent_vecs[:,k] = temp/count                    # New cluster centroids computed

	for i in range(xmax):
		for j in range(ymax):
			out_img[i,j,:] = cent_vecs[:,int(cent_vals[i,j])]  # Output image made of centroids created

	diff = out_img-img
	cost = np.linalg.norm(diff, axis=2)
	cost = np.sum(cost)                                        # Cost computed

	return out_img, cost                                       # Output image and cost returned

'''
************************************************************************************
K-means clustering:
************************************************************************************
'''

image = im.open('car.png')              # Source image opened as an 24-bit RGB intensity matrix
car = np.asarray(image)                 # Converted into numpy array
dim = np.shape(car)
xmax1 = dim[0]
ymax1 = dim[1]                          # Image dimensions are obtained

image = im.open('flower.png')           # Source image opened as an 24-bit RGB intensity matrix
flower = np.asarray(image)              # Converted into numpy array
dim = np.shape(flower)
xmax2 = dim[0]
ymax2 = dim[1]                          # Image dimensions are obtained

K = 3
num = 5
N = 30                                  # Parameters initialized as given in the assignment

ini_vecs = np.array([[255,0,255],[0,0,255],[0,0,255]])
car_out_fix = np.zeros([xmax1,ymax1,3], dtype = np.uint8)
flower_out_fix = np.zeros([xmax2,ymax2,3], dtype = np.uint8)     

car_out_fix = K_means(car,xmax1,ymax1,K,ini_vecs,num)[0]
flower_out_fix = K_means(flower,xmax2,ymax2,K,ini_vecs,num)[0]    # Fixed initialization outputs obtained

car_out_rand = np.zeros([xmax1,ymax1,3,N], dtype = np.uint8)
flower_out_rand = np.zeros([xmax2,ymax2,3,N], dtype = np.uint8)
car_costs = np.zeros(N)
flower_costs = np.zeros(N)

# For loop to run the randomly initialized K-means algorithm 
for index in range(N):
	ini_vecs = np.array(rand.sample(range(256),3*K))
	ini_vecs = np.reshape(ini_vecs,(3,K))

	car_out_rand[:,:,:,index], car_costs[index] = K_means(car,xmax1,ymax1,K,ini_vecs,num)
	flower_out_rand[:,:,:,index], flower_costs[index] = K_means(flower,xmax2,ymax2,K,ini_vecs,num)

car_maxind = np.argmax(car_costs)
car_minind = np.argmin(car_costs)
flower_maxind = np.argmax(flower_costs)                       # Indices corresponding to maximum and 
flower_minind = np.argmin(flower_costs)                       # minimum costs obtained          

car_out_rand_max = car_out_rand[:,:,:,car_maxind]
car_out_rand_min = car_out_rand[:,:,:,car_minind]
flower_out_rand_max = flower_out_rand[:,:,:,flower_maxind]    # Output images corresponding to maximum
flower_out_rand_min = flower_out_rand[:,:,:,flower_minind]    # and minimum costs obtained

car_out_fix = np.flip(car_out_fix, axis=2)
car_out_rand_max = np.flip(car_out_rand_max, axis=2)
car_out_rand_min = np.flip(car_out_rand_min, axis=2)
flower_out_fix = np.flip(flower_out_fix, axis=2)
flower_out_rand_max = np.flip(flower_out_rand_max, axis=2)
flower_out_rand_min = np.flip(flower_out_rand_min, axis=2)

cv2.imwrite("car_fix.png", car_out_fix)
cv2.imwrite("car_randmax.png", car_out_rand_max)
cv2.imwrite("car_randmin.png", car_out_rand_min)
cv2.imwrite("flower_fix.png", flower_out_fix)
cv2.imwrite("flower_randmax.png", flower_out_rand_max)
cv2.imwrite("flower_randmin.png", flower_out_rand_min)        # Output images are saved as .png files    