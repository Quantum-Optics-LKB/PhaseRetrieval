# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 28/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import EasyPySpin

Cam = EasyPySpin.VideoCapture(0)
fig=plt.figure(0)
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
while True:
    ret, frame = Cam.read()
    frame=cv2.flip(frame, 0)
    frame_blurred = cv2.blur(frame, (12,12))
    ret1, thresh = cv2.threshold(frame, 70, 255, 0)
    thresh_blurred = cv2.blur(thresh, (12, 12))
    ax1.imshow(thresh, cmap='gray')
    M = cv2.moments(thresh_blurred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    T = np.float32([[1, 0, 1024-cX], [0, 1, 1024-cY]])
    frame_s = cv2.warpAffine(frame, T, frame.shape)
    ax.imshow(frame_s, cmap='gray')
    #circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 50, minRadius=400, maxRadius=1100)
    #circles = np.round(circles[0, :]).astype(int)
    #for (x,y,r) in circles:
    #    circle = plt.Circle((x,y), r, fill=False, color='g')
    #    ax1.add_artist(circle)
    #    ax1.scatter(x,y,color='b', marker='.')
    ax.scatter(1024,1024, color='r', marker='.')
    ax1.scatter(1024,1024, color='r', marker='.')
    ax1.scatter(cX,cY, color='b', marker='.')
    plt.pause(0.01)
plt.show()
Cam.release()
cv2.destroyAllWindows()