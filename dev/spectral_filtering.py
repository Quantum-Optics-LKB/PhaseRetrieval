# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

size=24.46e-3
wv = 532e-9
f=2
R=10e-3

cwd = os.getcwd()
I_final = np.asarray(Image.open(f"{cwd}/results_antiring_3840_z60cm/I_final.png"))[:,:,0]
I_final=I_final/np.max(I_final)
phi_final = np.asarray(Image.open(f"{cwd}/results_antiring_3840_z60cm/phi_final.png"))[:,:,0]
phi_final = 2*np.pi*(phi_final/np.max(phi_final))-np.pi*np.ones(phi_final.shape)
#try to filter it with a lens and a pinhole
A=Begin(size, wv, I_final.shape[0])
A=SubIntensity(I_final, A)
A=SubPhase(phi_final, A)
A=Forvard(2*f, A)
A=Lens(f, 0, 0, A)
A=Forvard(f, A)
A=CircAperture(R, 0, 0, A)
A=Forvard(f, A)
I=np.reshape(Intensity(1, A), I_final.shape)
phi=np.reshape(Phase(A), I_final.shape)
phi_tf = np.fft.fft2(phi_final)
phi_tf = np.fft.fftshift(phi_tf)
fig=plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im1 = ax1.imshow(I)
im2 = ax2.imshow(np.abs(phi_tf))
plt.colorbar(im1, cax1)
plt.colorbar(im2, cax2)
plt.show()