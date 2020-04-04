# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image #for custom phase / intensity masks

size=4*cm
wavelength=781*nm
N=2048
w=20*mm
f=100*cm #focal length
z=2*f #propagation distance

def phase_retrieval(I0: np.ndarray, I: np.ndarray, f: float, N: int):
    """
    Assumes a 2f-2f setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :return: The calculated phase_map using Gerchberg-Saxton algorithm
    """
    for i in range(N):
        #initiate propagating field
        A=Begin()



#initiate custom phase and intensity filters emulating the SLM
phi0 = np.asarray(Image.open("harambe.bmp"))[:,:,2] #extract only the first channel
#phi0 = np.asarray(Image.open("calib.bmp"))
#print(np.max(phi0)

phi0= (phi0+128)*(2*np.pi/255) #conversion to rads
#apply SLM filter to initiate the field in the SLM plane
Field = Begin(size, wavelength, N)
Field=RectAperture(1*cm,1*cm,0,0,0,Field)
Field=SubPhase(phi0,Field)
I1=Intensity(2, Field)
phi1=Phase(Field)
#propagate to the lens
Field = Forvard(z, Field)
#apply lens filter
Field = Lens(f,0,0,Field)
#propagate to image plane
Field = Forvard(z, Field)
#Retrieve intensity and phase
I2 = Intensity(2,Field)
phi2 = Phase(Field)
#Plot intensities @ different points
fig = plt.figure(0)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(I1, cmap="gray"); ax1.set_title("Intensity @z=0")
ax2.imshow(I2, cmap="gray" ); ax2.set_title("Intensity @z=4f")
#plot phase @ different points
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
ax1.imshow(phi1, cmap="gray"); ax1.set_title("Phase @z=0")
ax2.imshow(phi2, cmap="gray" ); ax2.set_title("Phase @z=4f")


plt.show()
