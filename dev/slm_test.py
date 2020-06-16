# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 12/06/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import EasyPySpin
from SLM import SLMscreen

Cam = EasyPySpin.VideoCapture(0)
resX, resY = 1280, 1024
ims=np.zeros((2048,2048,100))

slm=SLMscreen(resX, resY)
for i in range(10):
    print(i)
    slm_pic=(256 * np.random.rand(resY, resX)).astype('uint8')
    slm.update(slm_pic)
    ret, frame = Cam.read()
    frame=cv2.flip(frame, 0)
    ims[:,:,i]=frame
Cam.release()
slm.close()

#slm_display.close()
for k in range(10):
    plt.figure()
    plt.imshow(ims[:,:,k])
    plt.show()