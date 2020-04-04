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
    """
    # initiate propagating field
    A = Begin(size, wavelength, N)
    # seed random phase
    A = SubPhase(np.random.rand(I.shape[0], I.shape[1]), A)
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
    """
    T0 = time()
    h, w = I0.shape
    #Initiate random phase map in SLM plane for starting point
    pm_s = np.random.rand(h, w)
    #Assume flat wavefront in image plane
    pm_f = np.ones((h, w))
    #Intensity in SLM plane is source intensity
    am_s = np.sqrt(I0)
    #Intensity in image plane is target intensity
    am_f = np.sqrt(I)
    #initiate field in SLM plane
    signal_s = am_s * np.exp(pm_s * 1j)

    for i in range(k):
        T1 = time()
        #propagate to image plane
        signal_f = np.fft.fft2(signal_s)
        #retrieve phase in image plane
        pm_f = np.angle(signal_f)
        #impose target intensity
        signal_f = am_f * np.exp(pm_f * 1j)
        #propagate back to SLM plane
        signal_s = np.fft.ifft2(signal_f)
        #retrieve phase in SLM plane
        pm_s = np.angle(signal_s)
        #Define new propagating field
        signal_s = am_s * np.exp(pm_s * 1j)
        T2 = time()- T1
        if k % 10 == 0:
            print(f"{round(100*(i/k), ndigits=3)} % done ... ({T2} s per step)")
    pm = pm_f
    T3 = time()-T0
    print(f"Elapsed time : {T3} s")
    return pm

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
phi3=phase_retrieval(I1, I2, f, 500)
#propagate the computed solution to image plane
A = Begin(size, wavelength, N)
A = RectAperture(0.5*cm,0.5*cm,0,0,0,A)
A = SubPhase(phi3, A)
A = Forvard(z, A)
A = Lens(f, 0, 0, A)
A = Forvard(z, A)
I3 = np.reshape(Intensity(2,A), (N,N))



#Plot intensities @ different points
fig = plt.figure(0)
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
ax1.imshow(I1, cmap="gray"); ax1.set_title("Intensity @z=0")
ax2.imshow(I2, cmap="gray" ); ax2.set_title("Intensity @z=4f")
ax3.imshow(I3, cmap="gray" ); ax3.set_title("Computed intensity from phase retrieval @z=4f")

#plot phase @ different points
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
ax1.imshow(phi1, cmap="gray"); ax1.set_title("Phase @z=0")
ax2.imshow(phi2, cmap="gray" ); ax2.set_title("Phase @z=4f")
ax3.imshow(phi3, cmap="gray" ); ax3.set_title("Retrieved phase")


plt.show()
