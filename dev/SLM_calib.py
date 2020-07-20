import numpy as np
import matplotlib.pyplot as plt
import EasyPySpin
import slmpy
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
    x = 640 - np.linspace(0,1279, 1280)
    y = 512 - np.linspace(0,1023, 1024)
    X,Y = np.meshgrid(x,y)
    out = np.zeros(X.shape, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii>(R-width/2)
    cond &= Radii < (R+width/2)
    out[cond] = value
    return out
def lens(f):
    """
    Generates a lens pattern of focal length f
    :param f: focal length in m
    :return: the phase mask corresponding to the lens
    """
    x = 12.5e-6*(640 - np.linspace(0, 1279, 1280))
    y = 12.5e-6*(512 - np.linspace(0, 1023, 1024))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    k=2*np.pi/780e-9
    phi = (k*R**2)/(2*f)
    phi = phi%(2*np.pi)
    phi/=2*np.pi
    out = (255*phi).astype('uint8')
    return out
L = lens(40e-3)
slm = slmpy.SLMdisplay(isImageLock=True)
#cam = EasyPySpin.VideoCapture(0)
G=grating(0,223,20)
C = circle(100, 20, 255)
slm.updateArray(C)
time.sleep(200000)
#Displays lenses
#for f in np.linspace(40e-3,45e-3, 40):
#    print(f)
#    L=lens(f)
#    slm.updateArray(L)
#    time.sleep(0.25)

#while True:
#    value = input("Pixel value ? ")
#    G = grating(0, value, 20)
    #C = circle(100, 20, value)
#    slm.updateArray(G)
#ret, frame = cam.read()
#frame = cv2.flip(frame, 0)
#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
slm.close()
#plt.imshow(frame)
#plt.show()
