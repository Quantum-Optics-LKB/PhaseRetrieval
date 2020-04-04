# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image #for custom phase / intensity masks

size=1*cm
wavelength=781*nm
N=1024
w=20*mm
f=10*cm #focal length
z=2*f #propagation distance
#initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("harambe.bmp"))[:,:,2] #extract only the first channel
phi0 = np.asarray(Image.open("calib.bmp"))
print(np.max(phi0))
#invert values
I0= 255*np.ones(I0.shape)-I0
phi0= 255*np.ones(phi0.shape)-phi0
#apply SLM filter to initiate the field in the SLM plane
Field = Begin(size, wavelength, N)
Field=SubIntensity(I0,Field)
Field=SubPhase(phi0,Field)
I1=Intensity(0, Field)
phi1=Phase(Field)
#propagate to the lens
Field = Forvard(z, Field)
I2=Intensity(0, Field)
phi2=Phase(Field)
#apply lens filter
Field = Lens(f,0,0,Field)
#propagate to image plane
Field = Forvard(z, Field)
#Retrieve intensity and phase
I3 = Intensity(0,Field)
phi3 = Phase(Field)
#Plot intensities @ different points
fig = plt.figure(0)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(I1, cmap="Greys"); ax1.set_title("Intensity @z=0")
ax2.imshow(I2, cmap="Greys" ); ax2.set_title("Intensity @z=2f")
ax3.imshow(I3, cmap="Greys"); ax3.set_title("Intensity @z=4f")
#plot phase @ different points
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
ax1.imshow(phi1, cmap="Greys"); ax1.set_title("Phase @z=0")
ax2.imshow(phi2, cmap="Greys" ); ax2.set_title("Phase @z=2f")
ax3.imshow(phi3, cmap="Greys"); ax3.set_title("Phase @z=4f")

plt.show()
