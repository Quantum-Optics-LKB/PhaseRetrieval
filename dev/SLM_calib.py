import numpy as np
import matplotlib.pyplot as plt
import EasyPySpin
import slmpy
from PIL import Image
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
def circle(R, width, value):
    """
    Draws a circle
    :param R: Radius of the circle
    :param width: Width of the circle in px
    :param value : Value inside the circle
    :return: The circle as a 2d array
    """
    x = 636 - np.linspace(0,1271, 1272)
    y = 512 - np.linspace(0,1023, 1024)
    X,Y = np.meshgrid(x,y)
    out = np.zeros(X.shape, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii>(R-width/2)
    cond &= Radii < (R+width/2)
    out[cond] = value
    return out
def lens(f, b):
    """
    Generates a lens pattern of focal length f
    :param f: focal length in m
    :param b : bit value (0 to 255)
    :return: the phase mask corresponding to the lens
    """
    x = 12.5e-6*(636 - np.linspace(0, 1271, 1272))
    y = 12.5e-6*(512 - np.linspace(0, 1023, 1024))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    k=2*np.pi/780e-9
    phi = (k*R**2)/(2*f)
    phi = phi%(2*np.pi)
    phi/=2*np.pi
    out = (b*phi).astype('uint8')
    return out
L = lens(43.7e-3, 206)
slm = slmpy.SLMdisplay(isImageLock=True)
#cam = EasyPySpin.VideoCapture(0)
G=grating(0,206,20)
C = circle(100, 20, 100)
#load SLM flatness correction
corr = np.array(Image.open("/home/tangui/Documents/SLM/deformation_correction_pattern/CAL_LSH0802200_780nm.bmp"))
#plt.imshow(((206/255)*(C+corr)%256).astype("uint8"))
#plt.show()

slm.updateArray(((206/255)*(C-corr)%256).astype("uint8"))
#slm.updateArray(C)
time.sleep(200000)
#Displays lenses
for f in np.linspace(60e-3,120e-3, 20):
    print(f)
    L=lens(f, 206)
    slm.updateArray(((206/255)*(L-corr)%256).astype("uint8"))
    time.sleep(0.25)
#for b in np.linspace(180,255, 60, dtype='uint8'):
#    print(b)
    #G=grating(0,b,20)
#    L = lens(42.7e-3, b)
#    slm.updateArray(L)
#    time.sleep(0.25)

#while True:
#    value = input("Pixel value ? ")
#    G = grating(0, value, 20)
#    C = circle(100, 20, value)
#    slm.updateArray(G)
#    slm.updateArray(((206 / 255) * (C + corr) % 256).astype("uint8"))
#ret, frame = cam.read()
#frame = cv2.flip(frame, 0)
#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
slm.close()
#plt.imshow(frame)
#plt.show()
