# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image  # for custom phase / intensity masks
import time
from scipy.ndimage import interpolation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import os
import configparser
import ast
import argparse
import textwrap
import sys


#argument parser
parser = argparse.ArgumentParser(prog='ComputeCGH',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent('''\
         Compute hologram yielding target intensity after propagation. Config file format
         can be found in the README.md at https://github.com/quantumopticslkb/phase_retrieval
         '''))
parser.add_argument("I", help="Path to target intensity", type=str)
parser.add_argument("I0", help="Path to source intensity", type=str)
parser.add_argument("cfg", help="Path to config file", type=str)
parser.add_argument("-phi0", help="Path to source phase profile", type=str)
parser.add_argument("-mask_sr", help="Path to signal region mask. Can also be 'adaptative' for an automatic mask at \
                                 each iteration", type=str)
parser.add_argument("-output", help='Path to results folder', type=str)
parser.add_argument("-s", help='Program runs silent without plots', action='store_true')
args = parser.parse_args()

#progress bar
def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress : [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, ndigits=1), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def main():
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
        :param k: Number of iterations for GS algorithm
        :param unwrap : Phase unwrapping at the end
        :param plot : Toggles plots
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
        elif kwargs["mask_sr"]=='adaptative':
            mask_sr = np.ones((h,w))
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
            I_f_old = np.reshape(Intensity(1, signal_f), (h,w))  # retrieve far field intensity
            #if adaptative mask option, update the mask
            if "mask_sr" in kwargs and kwargs["mask_sr"]=='adaptative':
                mask_sr = define_mask(mask_sr*I_f_old, threshold, False) #no plots
            signal_f = SubIntensity(I * mask_sr + I_f_old * mask_nr,
                                    signal_f)  # Substitute the measured far field into the field only in the signal region
            signal_s = Forvard(-z, signal_f)  # Propagate back to the near field
            # interpolate to source size
            signal_s = Interpol(size, h_0, 0, 0, 0, 1, signal_s)
            signal_s = SubIntensity(I0, signal_s)  # Substitute the measured near field into the field
            T2 = time.time() - T1
            #if i % 10 == 0:
            #    progress=round(100 * (i / k), ndigits=3)
            #    print(f"{progress} % done ... ({T2} s per step)")
                #indent progress bar
            progress=float((i+1)/k)
            update_progress(progress)
        pm_s = Phase(signal_s)

        if unwrap:
            pm_s = PhaseUnwrap(pm_s)
        pm_s = np.reshape(pm_s, (h, w))
        T3 = time.time() - T0
        print(f"Elapsed time : {T3} s")
        return pm_s, mask_sr
    #modulation
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
        #define a radial position matrix
        R = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                R[i,j]=np.sqrt((h/2 - i)**2 + (w/2 - j)**2)
        sig=sigma*max(h,w)
        G=np.exp(-R**2/(2*sig**2))
        I_gauss = I*G
        return I_gauss

    #get current working directory
    cwd_path=os.getcwd()
    if args.output :
        results_path=f"{args.output}"
    else:
    #creates a path for the results folder with creation time in the name for convenience in terms of multiple calculation
        results_path=f"generated_cgh_"+str(time.gmtime().tm_hour)+str(time.gmtime().tm_min)+str(time.gmtime().tm_sec)
    #if the folder doesn't already exist, create it
    if not(os.path.isdir(f"{results_path}")):
        try:
            os.mkdir(f"{cwd_path}/{results_path}")
        except OSError:
            print("I did not manage to create the specified results folder, maybe there is an error in the specified path ?")
            raise

    #initiate parser that reads the config file
    cfg_path = args.cfg
    conf=configparser.ConfigParser()
    conf.read(cfg_path)

    #List of hardcoded parameters to read from a config file

    size = float(conf["params"]["size"])  # size of the SLM window
    wavelength = float(conf["params"]["wavelength"])
    z = float(conf["params"]["z"]) # propagation distance
    N_gs = int(conf["params"]["N_gs"]) # number of GS iterations
    N_mod = int(conf["params"]["N_mod"]) # number of modulation steps
    mod_intensity=float(conf["params"]["mod_intensity"]) #modulation intensity
    SLM_levels = int(conf["params"]["SLM_levels"]) #number of SLM levels
    mask_threshold = float(conf['params']['mask_threshold']) #intensity threshold for the signal region
    elements=[] #list of optical elements
    for element in conf["setup"]:
        elements.append(ast.literal_eval(conf['setup'][element]))

    # initiate  intensities, phase and mask
    I = np.asarray(Image.open(args.I))
    if I.ndim==3:
        if not(args.s):
            print("Target intensity is a multi-level image, taking the first layer")
        I = I[:, :, 0] # extract only the first channel if needed
    if args.I0:
        I0 = np.asarray(Image.open(args.I0))
        if I0.ndim == 3:
            if not (args.s):
                print("Initial intensity is a multi-level image, taking the first layer")
            I0 = I0[:, :, 0]
    else :
        I0 = np.ones(I.shape)
    #apply gaussian profile
    I0=gaussian_profile(I0, 0.5)
    I=gaussian_profile(I, 0.5)
    #normalize intensities
    I = I/np.max(I)
    I0 = I0/np.max(I0)
    h, w = I.shape
    h_0, w_0 = I0.shape
    if h!=h_0 and not(args.s):
        print("Warning : Different target and initial intensity dimensions. Interpolation will be used")
    if h!=w:
        if not (args.s):
            print("Non square target intensity specified. Target intensity will be extended with zeros to be square.")
        L=max(h,w) #size of the square
        tmp=np.zeros((L,L))
        i=int(L/2-h/2)
        j=int(L/2+h/2)
        k = int(L / 2 - w / 2)
        l = int(L / 2 + w / 2)
        tmp[i:j,k:l]=I
        I=tmp
    if h_0!=w_0:
        if not (args.s):
            print("Non square source intensity specified. Source intensity will be extended with zeros to be square.")
        L=max(h_0,w_0) #size of the square
        tmp=np.zeros((L,L))
        i=int(L/2-h_0/2)
        j=int(L/2+h_0/2)
        k = int(L / 2 - w_0 / 2)
        l = int(L / 2 + w_0 / 2)
        tmp[i:j,k:l]=I0
        I0=tmp

    # signal region for the RMS
    rms_sr = np.ones((h, w))
    rms_sr[np.where(I == 0)[0], np.where(I == 0)[1]] = 0
    rms_sr[np.where(I > 0)[0], np.where(I > 0)[1]] = 1
    # signal region for the initial intensity. Used only for auto padding.
    rms0_sr = np.ones((h_0, w_0))
    rms0_sr[np.where(I0 == 0)[0], np.where(I0 == 0)[1]] = 0
    rms0_sr[np.where(I0 > 0)[0], np.where(I0 > 0)[1]] = 1
    # compute if there is a pad of size h/4 h_0/4 around I / I0, if not pad the images up to twice their sizes
    # The >0.025*h**2 means that if the number of non zero points in the border region is more than 10% of the total
    # number of points in the border region, we consider that the border region is filled and so needs to be enlarged.
    I_is_not_padded = np.sum(rms_sr[0:int(h / 4), :]) > (0.025 * h ** 2) or np.sum(rms_sr[:, 0:int(h / 4)]) > (
                0.025 * h ** 2) or np.sum(rms_sr[int(3 * h / 4):h, :]) > (0.025 * h ** 2) \
                      or np.sum(rms_sr[:, int(3 * h / 4):h]) > (0.025 * h ** 2)
    I0_is_not_padded = np.sum(rms0_sr[0:int(h_0 / 4), :]) > (0.025 * h_0 ** 2) or np.sum(
        rms0_sr[:, 0:int(h_0 / 4)]) > (
                               0.025 * h_0 ** 2) or np.sum(rms0_sr[int(3 * h_0 / 4):h_0, :]) > (0.025 * h_0 ** 2) \
                       or np.sum(rms0_sr[:, int(3 * h_0 / 4):h_0]) > (0.025 * h_0 ** 2)
    if I_is_not_padded:
        print("The target intensity is not padded. It will be padded to twice its size with zeros.")
        tmp = np.zeros((2 * h, 2 * h))
        tmp[int(2*h/4):int(3*2*h/4), int(2*h/4):int(3*2*h/4)]=I
        I=tmp
    if I0_is_not_padded:
        print("The source intensity is not padded. It will be padded to twice its size with zeros.")
        tmp = np.zeros((2 * h_0, 2 * h_0))
        tmp[int(2*h_0 / 4):int(3 *2* h_0 / 4), int(2*h_0 / 4):int(3 *2* h_0 / 4)] = I0
        I0 = tmp
    # refresh all sizes.
    h, w = I.shape
    h_0, w_0 = I0.shape
    # refresh rms signal region
    rms_sr = np.ones((h, w))
    rms_sr[np.where(I == 0)[0], np.where(I == 0)[1]] = 0
    rms_sr[np.where(I > 0)[0], np.where(I > 0)[1]] = 1
    # if the initial phase was supplied, assign it. If not flat wavefront.
    if args.phi0:
        phi0 = np.asarray(Image.open(args.phi0))
        if phi0.ndim == 3:
            if not (args.s):
                print("Initial phase is a multi-level image, taking the first layer")
            phi0 = phi0[:, :, 0]
    else:
        phi0 = np.zeros((h_0, w_0))
    h_phi0, w_phi0 = phi0.shape
    if h_phi0!=w_phi0:
        if not (args.s):
            print("Non square source phase specified. Source phase will be extended with zeros to be square.")
        L=max(h_phi0,w_phi0) #size of the square
        tmp=np.zeros((L,L))
        i=int(L/2-h_0/2)
        j=int(L/2+h_0/2)
        k = int(L / 2 - w_0 / 2)
        l = int(L / 2 + w_0 / 2 )
        tmp[i:j,k:l]=phi0
        phi0=tmp
    if h_0!=h_phi0 and not(args.s):
        print("Warning : Different initial phase and initial intensity dimensions. Interpolation will be used")
    #refresh all sizes.
    h_phi0, w_phi0 = phi0.shape
    #Conversion of the initial phase to rad
    if args.phi0:
        phi0 = ((SLM_levels/2)*np.ones(phi0.shape)-phi0) * (2 * np.pi / SLM_levels)

    # signal region for the phase
    phi0_sr = np.ones((h_phi0, w_phi0))  # signal region
    phi0_sr[np.where(I0 == 0)[0], np.where(I0 == 0)[1]] = 0
    phi0_sr[np.where(I0 > 0)[0], np.where(I0 > 0)[1]] = 1

    #define mask
    if args.mask_sr and args.mask_sr!='adaptative':
        mask_sr = np.asarray(Image.open(args.mask_sr))
        if mask_sr.ndim == 3:
            if not (args.s):
                print("Signal region is a multi-level image, taking the first layer")
            mask_sr = mask_sr[:, :, 0]
            # check if signal region size matches the target intensity
        if mask_sr.shape != I.shape:
            print("Error : Signal region size does not match target intensity size !")
            raise
    elif args.mask_sr=='adaptative':
        mask_sr='adaptative'


    #if only one modulation step, do the regular computation
    Phi, Mask = [], []
    if N_mod ==1:
        # phase retrieval
        if not(args.s):
            if args.mask_sr:
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, threshold=mask_threshold, mask_sr=mask_sr, phi0=phi0)
            else :
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, threshold=mask_threshold, phi0=phi0)
        elif args.s :
            if args.mask_sr:
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, plot=False, threshold=mask_threshold, mask_sr=mask_sr, phi0=phi0)
            else :
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, plot=False, threshold=mask_threshold, phi0=phi0)
        Phi.append(phi)
        Mask.append(mask_sr)
    else :
        # phase retrieval (run in silent mode for better speed)
        T0 = time.time()
        for i in range(N_mod):
            print(f"Modulation step {i+1} of {N_mod}")
            phi_m = phi0 + modulate(phi0, mod_intensity)
            #phi_m = phi0 + (2*np.pi/N_mod)*np.ones((h_0, w_0))
            if args.mask_sr:
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, plot=False, threshold=mask_threshold,
                                               mask_sr=mask_sr, phi0=phi_m)
            else:
                phi, mask_sr = phase_retrieval(I0, I, N_gs, False, plot=False, threshold=mask_threshold, phi0=phi_m)
            Phi.append(phi)
            Mask.append(mask_sr)
        T=time.time()-T0
        print(f"Modulation done. Time elapsed {T} s")
    Phi=np.array(Phi)
    #save this array for later processing
    np.save("Phi", Phi)
    Mask = np.array(Mask)
    phi=np.mean(Phi, axis=0)

    # propagate the computed solution to image plane
    A = Begin(size, wavelength, h_0)
    A = SubIntensity(I0, A)
    #A = SubPhase(phi-phi0, A) #add source beam phase
    A = SubPhase(phi, A) #add source beam phase
    A = Forvard(z, A)
    I_final = np.reshape(Intensity(0, A), (h_0, h_0))
    phi_final = np.reshape(Phase(A), (h_0, h_0))
    phi_final_cut = phi_final[int(h/2),:]
    #Compute FT of reconstructed intensity.
    I_tf = np.fft.fft2(I_final)
    I_tf = np.abs(np.fft.fftshift(I_tf))
    phi_tf = np.fft.fft2(phi)
    phi_tf = np.abs(np.fft.fftshift(phi_tf))
    freq = np.fft.fftfreq(h, d=size/h)
    #Compute RMS
    RMS=np.sqrt(np.mean(rms_sr*(I-I_final)**2))
    # Compute intensity conversion efficiency
    conv_eff = np.sum(rms_sr * I_final) / np.sum(I0)
    vmin=np.min(mask_sr*I0)
    vmax=np.max(mask_sr*I0)
    #save results
    plt.imsave(f"{results_path}/I0.png",I0, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.imsave(f"{results_path}/I.png",I, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.imsave(f"{results_path}/I_final.png",I_final, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.imsave(f"{results_path}/phi0.png",phi0, cmap='viridis')
    plt.imsave(f"{results_path}/phi.png",phi, cmap='viridis')
    plt.imsave(f"{results_path}/phi_final.png",phi_final, cmap='viridis')
    f_rms=open(f"{results_path}/RMS_intensity.txt", "w+")
    f_rms.write(f"RMS for the intensity is : {RMS}")
    f_rms.close()
    f_iconv = open(f"{results_path}/conv_eff.txt", "w+")
    f_iconv.write(f"Conversion efficiency in the signal region is : {conv_eff}")
    f_iconv.close()
    f_cfg = open(cfg_path)
    config=f_cfg.read()
    f_cfg.close()
    f_cfg = open(f"{results_path}/config.conf", "w+")
    f_cfg.write(config)
    f_cfg.close()
    # Plot results : intensity and phase
    #min and max intensities in the signal region for proper normalization
    if not(args.s):
        fig = plt.figure(0)
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes('right', size='5%', pad=0.05)
        im1=ax1.imshow(phi, cmap="viridis", vmin=-np.pi, vmax=np.pi)
        ax1.set_title(f"Mean reconstructed phase")
        fig.colorbar(im1, cax = cax1)
        im2=ax2.imshow(I, cmap="viridis", vmin=vmin, vmax=vmax)
        ax2.set_title("Target intensity")
        fig.colorbar(im2, cax = cax2)
        im3=ax3.imshow(I_final, cmap="viridis", vmin=vmin, vmax=vmax)
        #ax3.imshow(np.ones(rms_sr.shape)-rms_sr,cmap='Greys', alpha=0.4) #grey over non signal region
        ax3.text(8, 18, f"RMS = {round(RMS, ndigits=3)} CONV = {round(conv_eff, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
        ax3.set_title("Propagated intensity (with mean recontructed phase)")
        fig.colorbar(im3, cax = cax3)
        #extent=[min(freq), max(freq), min(freq), max(freq)]
        #im4 = ax4.imshow(phi_tf, cmap="viridis", extent=extent)
        im4 = ax4.imshow(phi_final, cmap="viridis")
        ax4.set_title("Phase after propagation")
        fig.colorbar(im4, cax=cax4)
        X = np.linspace(0, h-1, h)
        ax5.plot(X, phi_final_cut)
        ax5.set_title("Phase after propagation")
        ax5.set_xlabel("Horizontal index")
        ax5.set_ylabel("Phase in rad")
        plt.show()
if __name__ == "__main__":
	main()
