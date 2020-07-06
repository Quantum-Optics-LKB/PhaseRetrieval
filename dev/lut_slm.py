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

def grating(N_ref, N, d):
    G = N_ref*np.ones((1024, 1280), dtype='uint8')
    y = np.linspace(0,1023,1024)
    x = np.linspace(0,1279,1280)
    X,Y = np.meshgrid(x,y)
    G[X%d==0]=N
    return G

mask =np.zeros((1024,1280), dtype='uint8')
mask[:,0:640]=255
#for i in range(mask.shape[1]):
#    if i%2==0:
#      mask[:,i]=255
cam = EasyPySpin.VideoCapture(0)
slm_screen = SLMscreen(1280,1024)
N=10
maxis = np.empty(N)
for i in enumerate(np.linspace(0,255,N, dtype='uint8')):
    print(f"Displaying {i[0]+1}th mask")
    mask =grating(0,i[1],2)
    slm_screen.update(mask)
    ret, frame = cam.read()
    maxis[i[0]]=2*np.arcsin(np.sqrt(np.max(frame[0:960,:])/np.max(frame)))
m, b = np.polyfit(np.linspace(0,255,N, dtype='uint8'), maxis, 1)
slm_screen.close()
cam.release()
fig=plt.figure(1)
ax0=fig.add_subplot(111)
ax0.plot(np.linspace(0,255,N, dtype='uint8'),maxis)
ax0.plot(np.linspace(0,255,N, dtype='uint8'), m*np.linspace(0,255,N, dtype='uint8')+b, color='r')
ax0.text(8, 18, f"m = {m}, b={b}", bbox={'facecolor': 'white', 'pad': 4})
plt.show()

