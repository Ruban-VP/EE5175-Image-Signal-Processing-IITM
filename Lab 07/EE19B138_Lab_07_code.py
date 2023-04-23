'''
************************************************************************************
EE5175: Image signal processing - Lab 07 Python code
Author: V. Ruban Vishnu Pandian (EE19B138)
Date: 17/03/2023
Execution command: python3 EE19B138_Lab_07_code.py
Note: Ensure all the source images are present in the present working directory  
************************************************************************************
'''

import numpy as np
from PIL import Image as im
import math as ma
import matplotlib.pyplot as plt                       # Importing the required libraries

'''
************************************************************************************
2D DFT and IDFT functions:
************************************************************************************
'''

def two_D_DFT(inp,M,N):

	inp = np.fft.fftshift(inp, axes=(0,))             # Periodic shifting to make image matrix 
	inp = np.fft.fftshift(inp, axes=(1,))             # upper left corner as origin

	out = np.zeros([M,N], dtype=complex)
	for m in range(M):
		out[m,:] = np.fft.fft(inp[m,:])               # 1D FFT taken for each row

	for l in range(N):
		out[:,l] = np.fft.fft(out[:,l])               # 1D FFT taken for each column

	out = np.fft.fftshift(out, axes=(0,))
	out = np.fft.fftshift(out, axes=(1,))             # Periodic shifting to make DFT matrix center as origin

	return out

def two_D_IDFT(inp,M,N):

	inp = np.fft.fftshift(inp, axes=(0,))             # Periodic shifting to make DFT matrix
	inp = np.fft.fftshift(inp, axes=(1,))             # upper left corner as origin

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
DFT, Magnitude-Phase Dominance, and Rotation Property: 
************************************************************************************
'''

image = im.open('fourier.png').convert("L")           # Source image opened as an 8-bit grayscale intensity matrix
fourier = np.asarray(image)                           # Converted into numpy array

image = im.open('fourier_transform.png').convert("L") # Source image opened as an 8-bit grayscale intensity matrix
transform = np.asarray(image)                         # Converted into numpy array

dim = np.shape(transform)
M = dim[0]
N = dim[1]                                            # Image dimensions are obtained

DFT_fourier = two_D_DFT(fourier,M,N)
DFT_transform = two_D_DFT(transform,M,N)              # DFT output matrices obtained for the given images

# Magnitude-Phase mixing is done as asked in the assignment 
I3_DFT = np.multiply(np.abs(DFT_fourier),np.exp(1j*np.angle(DFT_transform)))
I4_DFT = np.multiply(np.abs(DFT_transform),np.exp(1j*np.angle(DFT_fourier)))

# Images are reconstructed by using IDFT on mixed DFT matrices
I3 = two_D_IDFT(I3_DFT,M,N)
I4 = two_D_IDFT(I4_DFT,M,N) 

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(np.abs(DFT_fourier),cmap='gray')
plt.title("Fourier DFT magnitude")                       
plt.subplot(1,2,2)
plt.imshow(np.angle(DFT_fourier),cmap='gray')         
plt.title("Fourier DFT phase")                        # DFT of first image plotted

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(np.abs(DFT_transform),cmap='gray') 
plt.title("Transform DFT magnitude")                       
plt.subplot(1,2,2)
plt.imshow(np.angle(DFT_transform),cmap='gray')       
plt.title("Transform DFT phase")                      # DFT of second image plotted

plt.figure(3)
plt.imshow(np.abs(I3),cmap='gray')  
plt.title("Fourier magnitude with transform phase")   # Third image plotted

plt.figure(4)
plt.imshow(np.abs(I4),cmap='gray')                    
plt.title("Fourier phase with transform magnitude")   # Fourth image plotted

data = im.fromarray(np.abs(I3))
data = data.convert("L")
data.save('Image_3.png')

data = im.fromarray(np.abs(I4))
data = data.convert("L")
data.save('Image_4.png')                              # Image arrays are saved as .png files

image = im.open('peppers_small.png').convert("L")     # Source image opened as an 8-bit grayscale intensity matrix
pepper = np.asarray(image)                            # Converted into numpy array

dim = np.shape(pepper)
M = dim[0]
N = dim[1]                                            # Image dimensions are obtained

plt.figure(5)
plt.imshow(pepper,cmap='gray')
plt.title("Peppers image")                            # Image plotted      

pepper = np.fft.fftshift(pepper, axes=(0,))           # Periodic shifting to make image matrix
pepper = np.fft.fftshift(pepper, axes=(1,))           # upper left corner as origin 
			
DFT_pepper = np.zeros([M,N], dtype=complex)

inds = [i for i in range(N)]
Y,X = np.meshgrid(inds,inds)                         

for k in range(M):                                    # For loop to compute the rotated 2D DFT
	for l in range(N):
		temp = np.exp(-1j*2*ma.pi*(l*X-k*Y)/N)
		DFT_pepper[k,l] = np.sum(np.multiply(pepper,temp))

DFT_pepper = np.fft.fftshift(DFT_pepper, axes=(0,))
DFT_pepper = np.fft.fftshift(DFT_pepper, axes=(1,))   # Periodic shifting to make DFT matrix center as origin

pepper_recons = two_D_IDFT(DFT_pepper,M,N)            # 2D IDFT computed to reconstruct the image from rotated DFT

plt.figure(6)
plt.imshow(np.abs(pepper_recons),cmap='gray') 
plt.title("Rotated peppers image")                    # Reconstructed image plotted      

plt.show()

data = im.fromarray(np.abs(pepper_recons))
data = data.convert("L")
data.save('Rotated_Peppers.png')                      # Image saved as .png file