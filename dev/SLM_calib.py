import numpy as np
import matplotlib.pyplot as plt
import EasyPySpin
from SLM import SLMscreen
from PIL import Image
import time
import cv2

#Code to calibrate an SLM displaying a grating and fitting the fringes.
#create the grating
resX, resY = 1920, 1080
pxpitch = 8.0e-6  # SLM pixel pitch

def grating(N_ref, N, d):
    """
    Displays a grating of pitch d
    :param N_ref: Pixel value of the "blanks" of the grating
    :param N: Pixel value of the grating lines
    :param d: pitch
    :return: The grating as a 2d array
    """
    G = N_ref*np.ones((resY, resX), dtype='uint8')
    y = np.linspace(0, resY-1, resY)
    x = np.linspace(0, resX-1, resX)
    X, Y = np.meshgrid(x, y)
    G[X % d < d/2] = N
    return G
def circle(R, width, value):
    """
    Draws a circle
    :param R: Radius of the circle
    :param width: Width of the circle in px
    :param value : Value inside the circle
    :return: The circle as a 2d array
    """
    x = resX//2 - np.linspace(0, resX-1, resX)
    y = resY//2 - np.linspace(0, resY-1, resY)
    X, Y = np.meshgrid(x, y)
    out = np.zeros(X.shape, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii > (R-width/2)
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
    x = pxpitch*(resX//2 - np.linspace(0, resX-1, resX))
    y = pxpitch*(resY//2 - np.linspace(0, resY-1, resY))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    k = 1.5*2*np.pi/780e-9
    phi = (k*R**2)/(2*f)
    phi = phi % (2*np.pi)
    phi /= 2*np.pi
    out = (b*phi).astype('uint8')
    return out


L = lens(95.6e-3, 255)
slm = SLMscreen(resX, resY)
#cam = EasyPySpin.VideoCapture(0)
G = grating(0, 206, 20)
C = circle(100, 20, 128)
#load SLM flatness correction
# corr = np.array(Image.open("/home/tangui/Documents/SLM/deformation_correction_pattern/CAL_LSH0802200_780nm.bmp"))
corr = Image.open("/home/tangui/Documents/phase_retrieval/dev/phases/U14-2048-201133-06-04-07_808nm.png")
# corr = np.zeros((resY, resX))
#plt.imshow(((206/255)*(C+corr)%256).astype("uint8"))
#plt.show()

# slm.update(((250/255)*(L-corr) % 256).astype("uint8"))
slm.update(((250/255)*(C-corr) % 256).astype("uint8"))

# slm.update(C)
time.sleep(200000)
#Displays lenses
for f in np.linspace(94e-3, 96e-3, 20):
    print(f)
    L = lens(f, 255)
    slm.update(((250/255)*(L-corr) % 256).astype("uint8"))
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
