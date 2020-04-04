# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image #for custom phase / intensity masks
from time import time

size=4*cm
wavelength=781*nm
N=1024
w=20*mm
f=100*cm #focal length
z=2*f #propagation distance

def phase_retrieval(I0: np.ndarray, I: np.ndarray, f: float, k: int):
    """
    Assumes a 2f-2f setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :return phi: The calculated phase map using Gerchberg-Saxton algorithm
    """
    # initiate propagating field
    A = Begin(size, wavelength, N)
    T0 = time()
    for i in range(k):
        T1 = time()
        # Impose intensity
        A = SubIntensity(I, A)
        #Propagate backwards to lens
        A=Forvard(-z, A)
        #Apply lens
        A=Lens(f, 0, 0, A)
        #Propagate backwards to SLM plane
        A=Forvard(-z, A)
        #Extract phase
        phi=Phase(A)
        #define new field at SLM
        A = SubIntensity(I0, A)
        A = SubPhase(phi, A)
        # Propagate forward to lens
        A = Forvard(z, A)
        # Apply lens
        A = Lens(f, 0, 0, A)
        # Propagate forward to image plane
        A = Forvard(z, A)
        phi1 = Phase(A)
        T = time()-T1
        #print(f"Did a step ! It took me {T} seconds")
        if i%10==0:
            print(f"{100*(i/float(k))} % done")
    T=time()-T0
    print(f"It took me {T} seconds")
    Phi = np.reshape(phi, (N,N))
    return Phi
#initiate custom phase and intensity filters emulating the SLM
phi0 = np.asarray(Image.open("calib_1024.bmp")) #extract only the first channel
#phi0 = np.asarray(Image.open("calib.bmp"))
#print(np.max(phi0)

phi0= (phi0+128)*(2*np.pi/255) #conversion to rads
#apply SLM filter to initiate the field in the SLM plane
Field = Begin(size, wavelength, N)
Field=RectAperture(0.5*cm,0.5*cm,0,0,0,Field)
Field=SubPhase(phi0,Field)
I1=np.reshape(Intensity(2, Field), (N,N))
phi1=Phase(Field)
#propagate to the lens
Field = Forvard(z, Field)
#apply lens filter
Field = Lens(f,0,0,Field)
#propagate to image plane
Field = Forvard(z, Field)
#Retrieve intensity and phase
I2 = np.reshape(Intensity(2,Field), (N,N))
phi2 = Phase(Field)

#phase retrieval
phi3=phase_retrieval(I1, I2, f, 50)




#Plot intensities @ different points
#fig = plt.figure(0)
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#ax1.imshow(I1, cmap="gray"); ax1.set_title("Intensity @z=0")
#ax2.imshow(I2, cmap="gray" ); ax2.set_title("Intensity @z=4f")
#plot phase @ different points
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
ax1.imshow(phi1, cmap="gray"); ax1.set_title("Phase @z=0")
ax2.imshow(phi2, cmap="gray" ); ax2.set_title("Phase @z=4f")
ax3.imshow(phi3, cmap="gray" ); ax3.set_title("Retrieved phase")


plt.show()
