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

N = 512
size = N*6.4 * um  # size of the SLM window
wavelength = 532 * nm
f = 1 * cm  # focal length
z = 25*mm  # propagation distance
N_mod = 10  # number of modulated samples for phase retrieval
mod_intensity=0.1

def phase_retrieval(I0: np.ndarray, I: np.ndarray, k: int, unwrap: bool, plot:bool = False, threshold:float =1e-2,**kwargs):
    """
    Assumes a 2f-2f setup to retrieve the phase from the intensity at the image plane
    :param I0: Source intensity field
    :param I: Intensity field from which to retrieve the phase
    :param f: Focal length of the lens conjugating the two planes
    :param N: Number of iterations for GS algorithm
    :param unwrap : Phase unwrapping at the end
    :param threshold : Threshold for automatic mask float in [0,1] default is 1e-2
    :param **mask_sr : Signal region  np.ndarray
    :param **mask_nr : Noise region  np.ndarray
    :return phi: The calculated phase map using Gerchberg-Saxton algorithm
    """
    h, w = I0.shape
    mask_sr = np.zeros((h, w))
    #if no masks are specified, the function defines one
    if "mask_sr" not in kwargs:
        #detect outermost non zero target intensity point
        non_zero=np.array(np.where(I>threshold))
        non_zero_offset=np.zeros(non_zero.shape)
        #offset relative to center
        non_zero_offset[0]=non_zero[0]-(h/2)*np.ones(len(non_zero[0]))
        non_zero_offset[1] =non_zero[1]-(w / 2) * np.ones(len(non_zero[1]))
        #Determine radii of each non-zero point
        R_non_zero = np.sqrt(non_zero_offset[0]**2 + non_zero_offset[1]**2)
        R_max=np.where(R_non_zero==np.max(abs(R_non_zero)))[0][0] #if there are several equally far points, it takes the
                                                                  # first one
        i_max, j_max = int(h/2 + int(abs(non_zero_offset[0][R_max]))), int(w/2 + int(abs(non_zero_offset[1][R_max])))
        i_min, j_min = int(h/2 - int(abs(non_zero_offset[0][R_max]))), int(w/2 - int(abs(non_zero_offset[1][R_max])))
        delta_i=int(i_max-i_min)
        delta_j=int(j_max-j_min)
        if delta_i>delta_j:
            mask_sr[i_min:i_max, i_min:i_max]=1
        else :
            mask_sr[j_min:j_max, j_min:j_max] = 1
        mask_nr=np.ones(mask_sr.shape)-mask_sr
        if plot:
            fig=plt.figure(0)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(I, cmap="gray")
            ax1.set_title("Target intensity")
            ax2.imshow(mask_sr, cmap="gray")
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max],non_zero[1][R_max], color='r')
            scat.set_label('Threshold point')
            plt.legend()
            plt.show()

    T0 = time()
    signal_f = Begin(size, wavelength, h)
    I_f_old=np.zeros(I.shape)
    for i in range(k):
        T1 = time()
        signal_f = SubIntensity(I*mask_sr+I_f_old*mask_nr, signal_f)  # Substitute the measured far field into the field only in the signal region
        signal_s = Forvard(-z, signal_f)  # Propagate back to the near field
        signal_s = SubIntensity(I0, signal_s)  # Substitute the measured near field into the field
        signal_f = Forvard(z, signal_s)  # Propagate to the far field
        I_f_old= Intensity(0, signal_f) # retrieve far field intensity
        T2 = time() - T1
        if i % 10 == 0:
            print(f"{round(100 * (i / k), ndigits=3)} % done ... ({T2} s per step)")
    pm_s = Phase(signal_s)
    if unwrap :
        pm_s = PhaseUnwrap(pm_s)
    pm_s = np.reshape(pm_s, (h, w))
    T3 = time() - T0
    print(f"Elapsed time : {T3} s")
    return pm_s, mask_sr


def modulate(phi: np.ndarray, x: float):
    """
    A function to randomly modulating a phase map without introducing too much high frequency noise
    :param phi: Phase map to be modulated
    :param x : Modulation intensity. Must be between 0 and 1.
    :return: phi_m a modulated phase map to multiply to phi
    """
    # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
    h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
    M = np.pi*(x * (np.ones((h, w)) - 2*np.random.rand(h, w, ))) #random matrix between [-x*pi and x*pi]
    phi_m = interpolation.zoom(M, phi.shape[0] / h)
    phi_m = phi_m*np.pi #bring phase between [-pi.pi]
    return phi_m


# initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("harambe_512.bmp"))[:, :, 0]  # extract only the first channel
phi0_sr = np.ones((N,N)) #signal region
phi0_sr[np.where(I0==0)[0], np.where(I0==0)[1]]=0
phi0_sr[np.where(I0>0)[0], np.where(I0>0)[1]]=1
phi0_nr=np.ones((N,N))-phi0_sr
I0=np.ones((N,N))*phi0_sr
#for i in range(N):
#    for j in range(N):
#        I0[i,j]=np.exp(-(1/N**2)*((N/2-i)**2+(N/2-j)**2))
phi0 = np.asarray(Image.open("calib_512.bmp"))

phi0 = (128*np.ones(phi0.shape)-phi0) * (2 * np.pi / 256)# conversion to rads
phi0 = phi0_sr*phi0 #gating the phase to the signal region.
Phi_init = []
I_init = []
I_inter = []
Phi_final = []
Phi_m = []
I_final = []
Masks=[]
#define the 0 sum random patterns
for i in range(int(N_mod/2)):
    phi_m=modulate(phi0, mod_intensity)
    Phi_m.append(phi_m)
    Phi_m.append(-phi_m)
# modulation sequence : modulate with N_mod random SLM phase masks
for i in range(N_mod):
    print(f"Modulation step {i+1} of {N_mod}" )
    # apply SLM filter to initiate the field in the SLM plane
    Field = Begin(size, wavelength, N)
    Field = SubIntensity(I0, Field)
    #phi_m = np.zeros((N,N))
    phi_m=Phi_m[i]
    phi = phi0_sr*(phi_m+phi0)
    Phi_init.append(phi)
    Field = SubPhase(phi, Field)
    I_init.append(I0)
    # propagate to the sensor plane
    Field = Forvard(z, Field)
    #Field = Lens(f, 0, 0, Field)
    #Field = Forvard(z, Field)
    # Retrieve intensity and phase
    I1 = np.reshape(Intensity(1, Field), (N, N))
    I_inter.append(I1)
    # phase retrieval
    phi3, mask_sr = phase_retrieval(I0, I1, 200, False, threshold=5e-2)
    Phi_final.append(phi3-phi_m)
    Masks.append(mask_sr)
    # propagate the computed solution to image plane
    A = Begin(size, wavelength, N)
    A = SubIntensity(I0, A)
    A = SubPhase(phi3, A)
    A = Forvard(z, A)
    I_final.append(np.reshape(Intensity(0, A), (N, N)))
Phi_init = np.array(Phi_init)
Phi_final = np.array(Phi_final)
I_final = np.array(I_final)
I_inter = np.array(I_inter)
Masks = np.array(Masks)
# Average out the modulations to get final result
I_inter_m = np.mean(I_inter, axis=0)
Phi = np.mean(Phi_final, axis=0)
#define mean signal region for RMS computing
mask_sr=np.mean(Masks, axis=0)
A = Begin(size, wavelength, N)
A = SubIntensity(I0, A)
A = SubPhase(Phi, A)
A = Forvard(z, A)
I=(np.reshape(Intensity(0, A), (N, N)))
#target intensity
A = Begin(size, wavelength, N)
A = SubIntensity(I0, A)
A = SubPhase(phi0, A)
A = Forvard(z, A)
I_target=(np.reshape(Intensity(0, A), (N, N)))
#Compute RMS
RMS=(1/2*np.pi)*np.sqrt(np.mean(phi0_sr*(Phi-phi0)**2))
RMS_1=np.sqrt(np.mean(mask_sr*(I-I_inter_m)**2))
#compute correlation
corr=True
if corr:
    print("Computing phase correlation")
    T0=time()
    Corr=np.corrcoef(phi0_sr*phi0, phi0_sr*Phi)
    T1=time()-T0
    print(f"Done ! It took me {T1} s. Mean correlation is {np.mean(Corr)}")
# Plot results : intensity and phase
#min and max intensities in the signal region for proper normalization
vmin=np.min(mask_sr*I_target)
vmax=np.max(mask_sr*I_target)
fig = plt.figure(0)
if corr:
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
else :
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes('right', size='5%', pad=0.05)
im1=ax1.imshow(Phi, cmap="gray", vmin=-np.pi, vmax=np.pi)
ax1.set_title(f"Mean reconstructed phase")
ax1.text(8, 18, f"RMS = {round(RMS, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
fig.colorbar(im1, cax = cax1)
im2=ax2.imshow(I_target, cmap="gray", vmin=vmin, vmax=vmax)
ax2.set_title("Target intensity")
fig.colorbar(im2, cax = cax2)
im3=ax3.imshow(I, cmap="gray", vmin=vmin, vmax=vmax)
ax3.text(8, 18, f"RMS = {round(RMS_1, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
ax3.set_title("Propagated intensity (with mean recontructed phase)")
fig.colorbar(im3, cax = cax3)
if corr:
    im4=ax4.imshow(Corr, cmap="gray")
    ax4.set_title("Correlation between target phase and reconstructed phase")
    fig.colorbar(im4, cax = cax4)

"""
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3= fig1.add_subplot(133)
divider = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)
divider3 = make_axes_locatable(ax3)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
cax3 = divider3.append_axes('right', size='5%', pad=0.05)
im1=ax1.imshow((1/2*np.pi)*phi0, cmap="gray")
ax1.set_title("Initial phase")
im2=ax2.imshow(I0, cmap="gray")
ax2.set_title("Initial intensity")
im3=ax3.imshow(I_inter_m, cmap='gray')
ax3.set_title("Mean intensity @z")
fig1.colorbar(im1, cax = cax1)
fig1.colorbar(im2, cax = cax2)
fig1.colorbar(im3, cax = cax3)
"""
plt.show()
