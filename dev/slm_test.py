# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 12/06/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import EasyPySpin
from SLM import SLMscreen
import slmpy
import time
import sys

#Cam = EasyPySpin.VideoCapture(0)
resX, resY = 1280, 1024
#ims=np.zeros((2048,2048,100))

#slm=SLMscreen(resX, resY)
slm = slmpy.SLMdisplay(isImageLock = True)
T = np.zeros(200)
for i in range(200):
    sys.stdout.flush()
    slm_pic=(256 * np.random.rand(resY, resX)).astype('uint8')
    t0 = time.time()
    slm.updateArray(slm_pic)
    t=time.time()-t0
    T[i]=t
    sys.stdout.write(f"\r{i} : time displayed = {t} s")
    #ret, frame = Cam.read()
    #frame=cv2.flip(frame, 0)
    #ims[:,:,i]=frame
#Cam.release()
slm.close()
print(f"\nAverage display time = {np.mean(T)} ({np.std(T)}) s")
#slm_display.close()
#for k in range(10):
#    plt.figure()
#    plt.imshow(ims[:,:,k])
#    plt.show()