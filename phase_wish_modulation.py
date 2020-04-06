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

size = 4 * cm
wavelength = 781 * nm
N = 1024
w = 36.9 * mm  # size of the SLM window
f = 100 * cm  # focal length
z = f  # propagation distance
N_mod = 10  # number of modulated samples for phase retrieval


def phase_retrieval(I0: np.ndarray, I: np.ndarray, k: int):
    """
    Assumes a 2f-2f setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :return phi: The calculated phase map using Gerchberg-Saxton algorithm
    """

    T0 = time()
    h, w = I0.shape
    # Initiate random phase map in SLM plane for starting point
    pm_s = np.random.rand(h, w)
    # Assume flat wavefront in image plane
    pm_f = np.ones((h, w))
    # Intensity in image plane is target intensity
    am_f = np.sqrt(I)
    # Intensity in SLM plane is target intensity
    am_s = np.sqrt(I0)
    # initiate field in SLM plane
    signal_s = am_s * np.exp(pm_s * 1j)

    for i in range(k):
        T1 = time()
        # propagate to image plane
        signal_f = np.fft.fft2(signal_s)
        # retrieve phase in image plane
        pm_f = np.angle(signal_f)
        # impose target intensity
        signal_f = am_f * np.exp(pm_f * 1j)
        # propagate back to SLM plane
        signal_s = np.fft.ifft2(signal_f)
        # retrieve phase in SLM plane
        pm_s = np.angle(signal_s)
        # Define new SLM field field by imposing source intensity
        signal_s = am_s * np.exp(pm_s * 1j)
        T2 = time() - T1
        if i % 10 == 0:
            print(f"{round(100 * (i / k), ndigits=3)} % done ... ({T2} s per step)")
    pm = pm_f
    T3 = time() - T0
    print(f"Elapsed time : {T3} s")
    return pm


def modulate(phi: np.ndarray):
    """
    A function to randomly modulating a phase map without introducing too much high frequency noise
    :param phi:
    :return: phi_m a modulated phase map
    """
    # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
    h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
    M = np.random.rand(h, w)
    phi_m = interpolation.zoom(M, phi.shape[0] / h)
    return phi_m * phi


# initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("harambe.jpg"))[:, :, 0] # extract only the first channel
phi0 = np.asarray(Image.open("calib_1024_full.bmp"))
# phi0 = np.asarray(Image.open("calib.bmp"))
# print(np.max(phi0)

phi0 = (phi0 + 128) * (2 * np.pi / 255)  # conversion to rads
Phi_init = []
I_init = []
I_inter = []
Phi_final = []
I_final = []
# modulation sequence : modulate with N_mod random SLM phase masks
for i in range(N_mod):
    # apply SLM filter to initiate the field in the SLM plane
    Field = Begin(size, wavelength, N)
    Field = SubIntensity(I0, Field)
    phi = modulate(phi0)
    Phi_init.append(phi)
    Field = SubPhase(phi, Field)
    I1 = np.reshape(Intensity(2, Field), (N, N))
    I_init.append(I1)
    # propagate to the captor plane
    Field = Forvard(z, Field)
    # Retrieve intensity and phase
    I2 = np.reshape(Intensity(2, Field), (N, N))
    I_inter.append(I2)
    # phase retrieval
    phi3 = phase_retrieval(I1, I2, 50)
    Phi_final.append(phi3)
    # propagate the computed solution to image plane
    A = Begin(size, wavelength, N)
    A = RectAperture(0.5 * cm, 0.5 * cm, 0, 0, 0, A)
    A = SubPhase(phi3, A)
    A = Forvard(z, A)
    A = Lens(f, 0, 0, A)
    A = Forvard(z, A)
    I_final.append(np.reshape(Intensity(2, A), (N, N)))
Phi_final = np.array(Phi_final)
I_final = np.array(I_final)
I_inter = np.array(I_inter)
# Average out the modulations to get final result
Phi = np.mean(Phi_final, axis=0)
I = np.mean(I_final, axis=0)
# Plot results : intensity and phase
fig = plt.figure(0)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(Phi, cmap="gray");
ax1.set_title("Mean reconstructed phase")
ax2.imshow(I, cmap="gray");
ax2.set_title("Mean propagated intensity")
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
ax1.imshow(phi0, cmap="gray");
ax1.set_title("Initial phase")
ax2.imshow(I0, cmap="gray");
ax2.set_title("Initial intensity")
plt.show()
