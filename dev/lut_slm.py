# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 18/06/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from SLM import SLMscreen
#import slmpy
import cv2
import EasyPySpin
import matplotlib.pyplot as plt
#plt.switch_backend('QT5Agg')


mask =np.zeros((1024,1280), dtype='uint8')
mask[:,0:640]=255
#for i in range(mask.shape[1]):
#    if i%2==0:
#      mask[:,i]=255
cam = EasyPySpin.VideoCapture(0)
slm_screen = SLMscreen(1280,1024)
slm_screen.update(mask)
ret, frame = cam.read()
slm_screen.close()
cam.release()
plt.figure(1)
plt.imshow(frame)
plt.show()