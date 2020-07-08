# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 28/05/2020
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import EasyPySpin
import SLM
from scipy import ndimage

def grating(N_ref, N, d):
    """
    Displays a grating of pitch d
    :param N_ref: Pixel value of the "blanks" of the grating
    :param N: Pixel value of the grating lines
    :param d: pitch
    :return: The grating as a 2d array
    """
    G = N_ref*np.ones((1024, 1280), dtype='uint8')
    y = np.linspace(0,1023,1024)
    x = np.linspace(0,1279,1280)
    X,Y = np.meshgrid(x,y)
    G[X%d < d/2]=N
    return G

def circle(R, width):
    """
    Draws a circle
    :param R: Radius of the circle
    :param width: Width of the circle in px
    :return: The circle as a 2d array
    """
    pass
Cam = EasyPySpin.VideoCapture(0)




slm = SLM.SLMscreen(1280,1024)
G=grating(0,223,20)
slm.update(G)
mpl.rcParams['toolbar'] = 'toolbar2'
mpl.rcParams['image.interpolation'] = 'None'
mpl.rcParams['image.resample'] = False
fig=plt.figure(1)
ax = fig.add_subplot(131)
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
#divider1 = make_axes_locatable(ax1)
#cax1 = divider1.append_axes('right', size='5%', pad=0.05)
frame_nbr=0
#for _ in range(1000):
    #print(frame_nbr)
    #frame_nbr+=1
ret, frame = Cam.read()
frame=cv2.flip(frame, 0)
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Convert the img to grayscale
th = 135
gray = (frame/(2**8)).astype('uint8')
gray[gray>th]=255
gray[gray<=th]=0
#Define ROI only in the center of image
gray[0:512,:]=0
gray[1536:2048,:]=0
gray[:,0:512]=0
gray[:,1536:2048]=0
# Apply edge detection method on the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# This returns an array of r and theta values
lines = cv2.HoughLines(edges, 1, np.pi/10000, 600)
# The below for loop runs till r and theta values
# are in the range of the 2d array
print(lines[:,0,:].shape)
thetas=np.empty(lines[:,0,:].shape[0])
i=0
for r, theta in lines[:,0,:]:
    #print((theta/(2*np.pi))*360)
    thetas[i]=(theta/(2*np.pi))*360
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    # x0 stores the value rcos(theta)
    x0 = a * r
    # y0 stores the value rsin(theta)
    y0 = b * r
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 100 * (-b))
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(gray.shape[0]/2+y0 + 100 * (a))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 100 * (-b))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(gray.shape[0]/2+y0 - 100 * (a))
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    ax1.imshow(gray)
    #ax1.plot([x1, x2], [y1,y2], 'ro-')
    i+=1
    #frame_blurred = cv2.blur(frame, (12,12))
#ret1, thresh = cv2.threshold(frame, 70, 255, 0)
#thresh_blurred = cv2.blur(thresh, (12, 12))
#im1 = ax1.imshow(thresh, cmap='gray')
#ax1.set_title("Thresholded image")
#M = cv2.moments(thresh_blurred)
#cX = int(M["m10"] / M["m00"])
#cY = int(M["m01"] / M["m00"])
#T = np.float32([[1, 0, 1024-cX], [0, 1, 1024-cY]])
#frame_s = cv2.warpAffine(frame, T, frame.shape)
im = ax.imshow(frame, cmap='gray')
ax.set_title("Sensor image")
scat=ax.scatter(1024,1024, color='r', marker='.')
ax.axvline(x=1024, color='red', linewidth=1)
ax.axhline(y=1024, color='red', linewidth=1)
#scat10=ax1.scatter(1024,1024, color='r', marker='.')
#scat11=ax1.scatter(cX,cY, color='b', marker='.')
scat.set_label('Sensor center')
#scat10.set_label('Sensor center')
#scat11.set_label('Detected centroid')
#if frame_nbr==1:
ax.legend()
    #ax1.legend()
plt.colorbar(im, cax)
    #plt.colorbar(im1, cax1)
#plt.pause(0.01)
fig2=plt.figure(2)
ax3=fig2.gca()
counts, bins =np.histogram(thetas, bins=50, range=(0,1))
ax3.hist(thetas, bins=50, range=(0,1))
ax3.set_xlabel("Angle in degrees")
ax3.set_ylabel("Number of lines at the angle")
theta=0.5*(bins[np.where(counts==np.max(counts))[0][0]+1]+bins[np.where(counts==np.max(counts))[0][0]])
print(f"Angle is {theta} °")
ax2.imshow(ndimage.rotate(frame, theta, reshape=False), cmap='gray')
ax2.set_title("Corrected image")
ax2.scatter(1024,1024, color='r', marker='.')
ax2.axvline(x=1024, color='red', linewidth=1)
ax2.axhline(y=1024, color='red', linewidth=1)
plt.show()
Cam.release()
cv2.destroyAllWindows()