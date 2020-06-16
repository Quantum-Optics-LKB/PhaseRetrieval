# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 28/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import EasyPySpin

Cam = EasyPySpin.VideoCapture(0)
fig=plt.figure(0)
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)

frame_nbr=0
for _ in range(1000):
    print(frame_nbr)
    frame_nbr+=1
    ret, frame = Cam.read()
    frame=cv2.flip(frame, 0)
    frame_blurred = cv2.blur(frame, (12,12))
    ret1, thresh = cv2.threshold(frame, 70, 255, 0)
    thresh_blurred = cv2.blur(thresh, (12, 12))
    im1 = ax1.imshow(thresh, cmap='gray')
    ax1.set_title("Thresholded image")
    M = cv2.moments(thresh_blurred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    T = np.float32([[1, 0, 1024-cX], [0, 1, 1024-cY]])
    frame_s = cv2.warpAffine(frame, T, frame.shape)
    im = ax.imshow(frame_s, cmap='gray')
    ax.set_title("Sensor image")
    scat=ax.scatter(1024,1024, color='r', marker='.')
    scat10=ax1.scatter(1024,1024, color='r', marker='.')
    scat11=ax1.scatter(cX,cY, color='b', marker='.')
    scat.set_label('Sensor center')
    scat10.set_label('Sensor center')
    scat11.set_label('Detected centroid')
    if frame_nbr==1:
        ax.legend()
        ax1.legend()
        plt.colorbar(im, cax)
        plt.colorbar(im1, cax1)
    plt.pause(0.01)
plt.show()
Cam.release()
cv2.destroyAllWindows()