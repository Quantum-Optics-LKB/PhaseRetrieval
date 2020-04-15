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


#size of the longest side of the source intensity
size_SLM = 24.46e-3
size = 24.46e-3
wavelength = 532e-9
z = 32e-2
N_gs = 200
#modulation number 1 is disabled.
N_mod = 1
mod_intensity=0.1
SLM_levels = 256
mask_threshold = 5e-2


def define_mask(I: np.ndarray, threshold: float, plot: bool):
    """
    A function to define the signal region automatically from the provided intensity and threshold
    :param I: intensity from which to define a signal region
    :param threshold: intensities below threshold are discarded
    :param plot: Plot or not the defined mask
    :return: mask_sr the defined mask
    """
    h, w = I.shape
    # compute FT
    I_tf = np.fft.fft2(I)
    I_tf = np.abs(np.fft.fftshift(I_tf))
    freq = np.fft.fftfreq(h, d=size / h)
    mask_sr = np.zeros((h, w))
    # detect outermost non zero target intensity point
    non_zero = np.array(np.where(I > threshold))
    non_zero_offset = np.zeros(non_zero.shape)
    # offset relative to center
    non_zero_offset[0] = non_zero[0] - (h / 2) * np.ones(len(non_zero[0]))
    non_zero_offset[1] = non_zero[1] - (w / 2) * np.ones(len(non_zero[1]))
    # Determine radii of each non-zero point
    R_non_zero = np.sqrt(non_zero_offset[0] ** 2 + non_zero_offset[1] ** 2)
    R_max = np.where(R_non_zero == np.max(abs(R_non_zero)))[0][
        0]
    # if there are several equally far points, it takes the
    # first one
    i_max, j_max = int(h / 2 + int(abs(non_zero_offset[0][R_max]))), int(
        w / 2 + int(abs(non_zero_offset[1][R_max])))
    i_min, j_min = int(h / 2 - int(abs(non_zero_offset[0][R_max]))), int(
        w / 2 - int(abs(non_zero_offset[1][R_max])))
    delta_i = int(i_max - i_min)
    delta_j = int(j_max - j_min)
    if delta_i > delta_j:
        mask_sr[i_min:i_max, i_min:i_max] = 1
    else:
        mask_sr[j_min:j_max, j_min:j_max] = 1
    if plot:
        fig = plt.figure(0)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        ax1.imshow(I, cmap="viridis")
        ax1.set_title("Source intensity and phase at z")
        ax2.imshow(mask_sr, cmap="viridis")
        ax2.set_title(f"Signal region (Threshold = {threshold})")
        scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max], color='r')
        scat.set_label('Threshold point')
        ax2.legend()
        extent = [min(freq), max(freq), min(freq), max(freq)]
        im3 = ax3.imshow(I_tf, cmap="viridis", extent=extent)
        ax3.set_title("Fourier transform of propagated intensity")
        fig.colorbar(im3, cax=cax3)
        plt.show()
    return mask_sr


def phase_retrieval(I0: np.ndarray, I: np.ndarray, k: int, unwrap: bool = False, plot: bool = True,
                    threshold: float = 1e-2, **kwargs):
    """
    Assumes the propagation in the provided setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :param unwrap : Phase unwrapping at the end
    :param threshold : Threshold for automatic mask float in [0,1] default is 1e-2
    :param **mask_sr : Signal region  np.ndarray
    :param **phi0 : Initial phase of the source np.ndarray
    :return phi: The calculated phase map using Gerchberg-Saxton algorithm
    """
    h_0, w_0 = I0.shape
    h, w = I.shape
    # initiate initial phase
    if "phi0" in kwargs:
        phi0 = kwargs["phi0"]
    else:
        phi0 = np.zeros((h, w))
    mask_sr = np.zeros((h, w))
    # initiate field in the SLM plane
    signal_s = Begin(size, wavelength, h_0)
    signal_s = SubIntensity(I0, signal_s)
    signal_s = SubPhase(phi0, signal_s)
    # propagate to image plane
    signal_f = Forvard(z, signal_s)
    # interpolate to target size
    signal_f = Interpol(size, h, 0, 0, 0, 1, signal_f)
    # Retrieve propagated intensity
    I_f = np.reshape(Intensity(1, signal_f), (h, w))
    # if no masks are specified, the function defines one
    if "mask_sr" not in kwargs:
        mask_sr = define_mask(I_f, threshold, plot)
    elif kwargs["mask_sr"] == 'adaptative':
        mask_sr = np.ones((h, w))
    else:
        mask_sr = kwargs["mask_sr"]
    mask_nr = np.ones(mask_sr.shape) - mask_sr
    T0 = time.time()
    # initiate field in the SLM plane
    signal_s = Begin(size, wavelength, h)
    signal_s = SubIntensity(I0, signal_s)
    signal_s = SubPhase(phi0, signal_s)
    for i in range(k):
        T1 = time.time()
        signal_f = Forvard(z, signal_s)  # Propagate to the far field
        # interpolate to target size
        signal_f = Interpol(size, h, 0, 0, 0, 1, signal_f)
        I_f_old = np.reshape(Intensity(1, signal_f), (h, w))  # retrieve far field intensity
        # if adaptative mask option, update the mask
        if "mask_sr" in kwargs and kwargs["mask_sr"] == 'adaptative':
            mask_sr = define_mask(mask_sr * I_f_old, threshold, False)  # no plots
        signal_f = SubIntensity(I * mask_sr + I_f_old * mask_nr,
                                signal_f)  # Substitute the measured far field into the field only in the signal region
        signal_s = Forvard(-z, signal_f)  # Propagate back to the near field
        # interpolate to source size
        signal_s = Interpol(size, h_0, 0, 0, 0, 1, signal_s)
        signal_s = SubIntensity(I0, signal_s)  # Substitute the measured near field into the field
        T2 = time.time() - T1
        # if i % 10 == 0:
        #    progress=round(100 * (i / k), ndigits=3)
        #    print(f"{progress} % done ... ({T2} s per step)")
        # indent progress bar
        progress = float((i + 1) / k)
        update_progress(progress)
    pm_s = Phase(signal_s)

    if unwrap:
        pm_s = PhaseUnwrap(pm_s)
    pm_s = np.reshape(pm_s, (h, w))
    T3 = time.time() - T0
    print(f"Elapsed time : {T3} s")
    return pm_s, mask_sr


# modulation
def modulate(phi: np.ndarray, x: float):
    """
    A function to randomly modulating a phase map without introducing too much high frequency noise
    :param phi: Phase map to be modulated
    :param x : Modulation intensity. Must be between 0 and 1.
    :return: phi_m a modulated phase map to multiply to phi
    """
    # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
    h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
    M = np.pi * (x * (np.ones((h, w)) - 2 * np.random.rand(h, w, )))  # random matrix between [-x*pi and x*pi]
    phi_m = interpolation.zoom(M, phi.shape[0] / h)
    phi_m = phi_m * np.pi  # bring phase between [-pi.pi]
    return phi_m


def gaussian_profile(I: np.ndarray, sigma: float):
    """

    :param I: Intensity to which a gaussian profile is going to be applied
    :param sigma: Standard deviation of the gaussian profile, in fraction of the provided intensity size
    :return: I_gauss : the "gaussianized" intensity
    """
    h, w = I.shape
    # define a radial position matrix
    R = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            R[i, j] = np.sqrt((h / 2 - i) ** 2 + (w / 2 - j) ** 2)
    sig = sigma * max(h, w)
    G = np.exp(-R ** 2 / (2 * sig ** 2))
    I_gauss = I * G
    return I_gauss


# initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("harambe_512.bmp"))[:, :, 0]  # extract only the first channel
phi0 = np.asarray(Image.open("calib_512.bmp"))


