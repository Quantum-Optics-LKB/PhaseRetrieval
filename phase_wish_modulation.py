# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image  # for custom phase / intensity masks
from time import time
from scipy.ndimage import interpolation
from mpl_toolkits.axes_grid1 import make_axes_locatable

size = 4*36.9 * mm  # size of the SLM window
wavelength = 781 * nm
N = 256
f = 10 * cm  # focal length
z = 73*mm  # propagation distance
N_mod = 10  # number of modulated samples for phase retrieval


def phase_retrieval(I0: np.ndarray, I: np.ndarray, k: int, unwrap: bool):
    """
    Assumes a 2f-2f setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :param unwrap : Phase unwrapping at the end
    :return phi: The calculated phase map using Gerchberg-Saxton algorithm
    """

    T0 = time()
    h, w = I0.shape
    # Assume flat wavefront in image plane
    # pm_f = np.ones((h, w))
    # Intensity in image plane is target intensity
    # am_f = np.sqrt(I)
    # Intensity in SLM plane is target intensity
    # am_s = np.sqrt(I0)
    # initiate field in SLM plane
    # signal_s = am_s * np.exp(pm_s * 1j)
    signal_s = Begin(size, wavelength, N)
    #signal_s = SubIntensity(I0, signal_s)
    #signal_s = SubPhase(pm_s, signal_s)

    for i in range(k):
        T1 = time()
        # propagate to image plane
        # signal_f = np.fft.fftshift(np.fft.fft2(signal_s, norm="ortho"))
        signal_f = Forvard(z, signal_s)
        #signal_f = Lens(f, 0, 0, signal_f)
        #signal_f = Forvard(z, signal_f)
        # retrieve phase in image plane
        pm_f = Phase(signal_f)
        # impose target intensity
        # signal_f = am_f * np.exp(pm_f * 1j)
        signal_f = SubIntensity(I, signal_f)
        # propagate back to SLM plane
        # signal_s = np.fft.ifft2(signal_f)
        signal_s = Forvard(-z, signal_f)
        #signal_s = Lens(f, 0, 0, signal_s)
        #signal_s = Forvard(-z, signal_s)
        # Define new SLM field field by imposing source intensity
        # signal_s = am_s * np.exp(pm_s * 1j)
        signal_s = SubIntensity(I0, signal_s)
        T2 = time() - T1
        if i % 10 == 0:
            print(f"{round(100 * (i / k), ndigits=3)} % done ... ({T2} s per step)")
    if unwrap :
        pm_f = PhaseUnwrap(pm_f)
    pm_f = np.reshape(np.array(Phase(signal_f)), (N, N))
    pm = pm_f
    T3 = time() - T0
    print(f"Elapsed time : {T3} s")
    return pm


def modulate(phi: np.ndarray, x: float):
    """
    A function to randomly modulating a phase map without introducing too much high frequency noise
    :param phi: Phase map to be modulated
    :param x : Modulation intensity. Must be between 0 and 1.
    :return: phi_m a modulated phase map to multiply to phi
    """
    # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
    h, w = int(phi.shape[0] / 2.5), int(phi.shape[1] / 2.5)
    M = x * (np.ones((h, w)) - 2*np.random.rand(h, w)) #random matrix between [-x and x]
    phi_m = interpolation.zoom(M, phi.shape[0] / h)
    phi_m = phi_m*np.pi #bring phase between [-pi.pi]
    return phi_m


# initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("harambe_256.bmp"))[:, :, 0]  # extract only the first channel
mask = np.ones((N,N))
mask[np.where(I0==0)[0], np.where(I0==0)[1]]=0
mask[np.where(I0>0)[0], np.where(I0>0)[1]]=1
#I0=np.ones((N,N))
#for i in range(N):
#    for j in range(N):
#        I0[i,j]=np.exp(-(1/N**2)*((N/2-i)**2+(N/2-j)**2))
phi0 = np.asarray(Image.open("calib_256.bmp"))
# phi0 = np.asarray(Image.open("calib.bmp"))
phi0 = (128*np.ones(phi0.shape)-phi0) * (2 * np.pi / 256)# conversion to rads
phi0 = mask*phi0 #gating the phase to the signal region.
Phi_init = []
I_init = []
I_inter = []
Phi_final = []
I_final = []
# modulation sequence : modulate with N_mod random SLM phase masks
for i in range(N_mod):
    print(f"Modulation step {i} of {N_mod}")
    # apply SLM filter to initiate the field in the SLM plane
    Field = Begin(size, wavelength, N)
    Field = SubIntensity(I0, Field)
    phi_m = modulate(phi0, 1)
    #phi_m = np.zeros((N,N))
    phi = phi_m+phi0
    Phi_init.append(phi)
    Field = SubPhase(phi, Field)
    I1 = np.reshape(Intensity(1, Field), (N, N))
    I_init.append(I1)
    # propagate to the captor plane
    Field = Forvard(z, Field)
    #Field = Lens(f, 0, 0, Field)
    #Field = Forvard(z, Field)
    # Retrieve intensity and phase
    I2 = np.reshape(Intensity(1, Field), (N, N))
    I_inter.append(I2)
    # phase retrieval
    phi3 = phase_retrieval(I1, I2, 4000, False)
    phi3=phi3-phi_m
    Phi_final.append(phi3)
    # propagate the computed solution to image plane
    A = Begin(size, wavelength, N)
    A = SubIntensity(I0, A)
    A = SubPhase(phi3, A)
    A = Forvard(z, A)
    I_final.append(np.reshape(Intensity(1, A), (N, N)))
Phi_init = np.array(Phi_init)
Phi_final = np.array(Phi_final)
I_final = np.array(I_final)
I_inter = np.array(I_inter)
# Average out the modulations to get final result
Phi = np.mean(Phi_final, axis=0)
I = np.mean(I_final, axis=0)
I_forvard = np.mean(I_inter, axis=0)
#Compute RMS
RMS=(1/2*np.pi)*np.sqrt(np.mean((mask*(Phi-phi0))**2))
# Plot results : intensity and phase
fig = plt.figure(0)
ax1 = fig.add_subplot(131)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
im1=ax1.imshow((1/2*np.pi)*Phi, cmap="gray")
ax1.set_title(f"Mean reconstructed phase RMS={RMS}")
fig.colorbar(im1, cax = cax)
im2=ax2.imshow(I_forvard, cmap="gray")
ax2.set_title("Mean propagated intensity")
im3=ax3.imshow(I, cmap="gray")
ax3.set_title("Mean propagated intensity (with recontructed phase)")
"""
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
ax2 = fig1.add_subplot(122)
im1=ax1.imshow((1/2*np.pi)*phi0, cmap="gray")
ax1.set_title("Initial phase")
fig.colorbar(im1, cax = cax)
im2=ax2.imshow(I0, cmap="gray")
ax2.set_title("Initial intensity")
"""
plt.show()
