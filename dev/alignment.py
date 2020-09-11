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
import slmpy
import time
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

def circle(R, width, value):
    """
    Draws a circle
    :param R: Radius of the circle
    :param width: Width of the circle in px
    :param value : Value inside the circle
    :return: The circle as a 2d array
    """
    x = 640 - np.linspace(0,1279, 1280)
    y = 512 - np.linspace(0,1023, 1024)
    X,Y = np.meshgrid(x,y)
    out = np.zeros(X.shape, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii>(R-width/2)
    cond &= Radii < (R+width/2)
    out[cond] = value
    return out
Cam = EasyPySpin.VideoCapture(0)
slm = slmpy.SLMdisplay(isImageLock = True)
G=grating(0,255,20)
C = circle(100, 20, 255)
slm.updateArray(G)
time.sleep(0.5)
ret, frame_l = Cam.read()
slm.updateArray(C)
time.sleep(0.5)
#frame_c = np.empty((10, 2048, 2048))
#for i in range(10):
    #ret, frame_c[i] = Cam.read()
ret, frame_c = Cam.read()
slm.close()
Cam.release()
frame_l=cv2.flip(frame_l, 0)
#frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_CLOCKWISE)
#frame_l = ndimage.rotate(frame_l, -0.228, reshape=False)
frame_c=cv2.flip(frame_c, 0)
#frame_c = cv2.rotate(frame_c, cv2.ROTATE_90_CLOCKWISE)
#frame_c = ndimage.rotate(frame_c, -0.228, reshape=False)
#plot captured images of lines and circles
fig=plt.figure(1)
ax0=fig.add_subplot(121)
ax1=fig.add_subplot(122)
ax0.imshow(frame_l)
ax1.imshow(frame_c)
plt.show()
# Convert the line img to binary grayscale
th = 85
gray_l = ((1/2**8)*frame_l).astype('uint8')
gray_l[gray_l>th]=255
gray_l[gray_l<=th]=0
#Define ROI only in the center of image
gray_l[0:256,:]=0
gray_l[768:1024,:]=0
gray_l[:,0:256]=0
gray_l[:,768:1024]=0
# convert the circle img to binary grayscale
th_c = 85
gray_c = ((1/2**8)*frame_c).astype('uint8')
gray_c[gray_c>th_c]=255
gray_c[gray_c<=th_c]=0
# Apply edge detection method on the image
edges_l = cv2.Canny(gray_l, 50, 150, apertureSize=3)
edges_c = cv2.Canny(gray_c, 50, 150, apertureSize=3)
# Line detection : This returns an array of r and theta values
lines = cv2.HoughLines(edges_l, 1, np.pi/10000, 300)
circles = cv2.HoughCircles(edges_c, cv2.HOUGH_GRADIENT, 1, 150)
# The below for loop runs till r and theta values
# are in the range of the 2d array
if np.any(lines!=None) :
    print(f"Detected {lines[:,0,:].shape[0]} lines")
    thetas = np.empty(lines[:, 0, :].shape[0])
    i = 0
    for r, theta in lines[:, 0, :]:
        # print((theta/(2*np.pi))*360)
        thetas[i] =180-(theta / (2 * np.pi)) * 360
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
        y1 = int(gray_l.shape[0] / 2 + y0 + 100 * (a))
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 100 * (-b))
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(gray_l.shape[0] / 2 + y0 - 100 * (a))
        # drawn. In this case, it is red.
        # ax1.plot([x1, x2], [y1,y2], 'r-')
        i += 1
else :
    print("No lines detected !")

if np.any(circles!=None):
    print(f"Detected {circles.shape[1]} circles")
else :
    print("No circles detected !")
fig=plt.figure(0)
ax0=fig.add_subplot(131)
ax1=fig.add_subplot(132)
ax2=fig.add_subplot(133)
ax0.imshow(frame_l)
ax1.imshow(frame_c)
ax2.hist(thetas, bins=3000)
counts, bins =np.histogram(thetas, bins=3000)
theta=0.5*(bins[np.where(counts==np.max(counts))[0][0]+1]+bins[np.where(counts==np.max(counts))[0][0]])
print(f"Angle is {theta} °")
ax2.set_xlabel("Angle in degrees")
ax2.set_ylabel("Number of lines detected at that angle")
for x,y,r in circles[0,:,:]:
    ax1.add_artist(plt.Circle((x,y), r, color='r', fill=False))
    ax1.scatter(x,y, color='r', marker='.')
    print(f"Coordinates of the circle's center : x={x}, y={y}")
plt.show()