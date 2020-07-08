import numpy as np
import matplotlib.pyplot as plt
import EasyPySpin
import SLM
import time
import cv2

#Code to calibrate an SLM displaying a grating and fitting the fringes.
#create the grating
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

slm = SLM.SLMscreen(1280,1024)
#cam = EasyPySpin.VideoCapture(0)
G=grating(0,223,20)
slm.update(G)
time.sleep(2000)
#ret, frame = cam.read()
#frame = cv2.flip(frame, 0)
#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
slm.close()
#plt.imshow(frame)
#plt.show()
