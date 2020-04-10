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
import os
import configparser
import ast
import argparse
import textwrap


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
parser.add_argument("-mask_sr", help="Path to signal region mask", type=str)
parser.add_argument("-output", help='Path to results folder', type=str)
parser.add_argument("-s", help='Program runs silent without plots', action='store_true')
args = parser.parse_args()

def main():
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
        :param **mask_nr : Noise region  np.ndarray
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
        if "mask_nr" not in kwargs:
            # detect outermost non zero target intensity point
            non_zero = np.array(np.where(I_f > threshold))
            non_zero_offset = np.zeros(non_zero.shape)
            # offset relative to center
            non_zero_offset[0] = non_zero[0] - (h / 2) * np.ones(len(non_zero[0]))
            non_zero_offset[1] = non_zero[1] - (w / 2) * np.ones(len(non_zero[1]))
            # Determine radii of each non-zero point
            R_non_zero = np.sqrt(non_zero_offset[0] ** 2 + non_zero_offset[1] ** 2)
            R_max = np.where(R_non_zero == np.max(abs(R_non_zero)))[0][
                0]  # if there are several equally far points, it takes the
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
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(I_f, cmap="gray")
                ax1.set_title("Source intensity and phase at z")
                ax2.imshow(mask_sr, cmap="gray")
                ax2.set_title(f"Signal region (Threshold = {threshold})")
                scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max], color='r')
                scat.set_label('Threshold point')
                plt.legend()
                plt.show()
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
            I_f_old = Intensity(0, signal_f)  # retrieve far field intensity
            signal_f = SubIntensity(I * mask_sr + I_f_old * mask_nr,
                                    signal_f)  # Substitute the measured far field into the field only in the signal region
            signal_s = Forvard(-z, signal_f)  # Propagate back to the near field
            # interpolate to source size
            signal_s = Interpol(size, h_0, 0, 0, 0, 1, signal_s)
            signal_s = SubIntensity(I0, signal_s)  # Substitute the measured near field into the field
            #pm_s = np.reshape(Phase(signal_s), (h_0, w_0))
            # signal_s = SubPhase(phi0+pm_s, signal_s) #add the source field phase
            T2 = time.time() - T1
            if i % 10 == 0:
                print(f"{round(100 * (i / k), ndigits=3)} % done ... ({T2} s per step)")
        pm_s = Phase(signal_s)
        if unwrap:
            pm_s = PhaseUnwrap(pm_s)
        pm_s = np.reshape(pm_s, (h, w))
        T3 = time.time() - T0
        print(f"Elapsed time : {T3} s")
        return pm_s, mask_sr

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
    #N_mod = int(conf["params"]["N_mod"])  # number of modulated samples for phase retrieval
    mod_intensity=float(conf["params"]["mod_intensity"]) #modulation intensity
    SLM_levels = int(conf["params"]["SLM_levels"]) #number of SLM levels
    mask_threshold = float(conf['params']['mask_threshold']) #intensity threshold for the signal region
    elements=[] #list of optical elements
    for element in conf["setup"]:
        elements.append(ast.literal_eval(conf['setup'][element]))
    # initiate  intensities, phase and mask
    I = np.asarray(Image.open(args.I))
    I0 = np.asarray(Image.open(args.I0))
    if I.ndim==3:
        if not(args.s):
            print("Target intensity is a multi-level image, taking the first layer")
        I = I[:, :, 0] # extract only the first channel if needed
    if I0.ndim==3:
        if not (args.s):
            print("Initial intensity is a multi-level image, taking the first layer")
        I0 = I0[:, :, 0]
    #normalize intensities
    I = I/np.max(I)
    I0 = I0/np.max(I0)
    h, w = I.shape
    h_0, w_0 = I0.shape
    # if the initial phase was supplied, assign it. If not flat wavefront.
    if args.phi0:
        phi0 = np.asarray(Image.open(args.phi0))
        if phi0.ndim == 3:
            if not (args.s):
                print("Initial phase is a multi-level image, taking the first layer")
            phi0 = phi0[:, :, 0]
        # check if initial phase size matches the source intensity
        if phi0.shape!=I0.shape:
            print("Error : Initial phase size does not match source intensity size !")
            raise
    else :
        phi0=np.zeros((h_0, w_0))

    if args.mask_sr:
        mask_sr = np.asarray(Image.open(args.mask_sr))
        if mask_sr.ndim == 3:
            if not (args.s):
                print("Signal region is a multi-level image, taking the first layer")
            mask_sr = mask_sr[:, :, 0]
            # check if signal region size matches the target intensity
        if mask_sr.shape != I.shape:
            print("Error : Signal region size does not match target intensity size !")
            raise

    if h!=h_0 and not(args.s):
        print("Warning : Different target and initial intensity dimensions. Interpolation will be used")
    if h!=w:
        if not (args.s):
            print("Non square target intensity specified. Target intensity will be extended with zeros to be square.")
        L=max(h,w) #size of the square
        tmp=np.zeros((L,L))
        i=int(L/2-h/2)
        j=int(L/2+h/2-1)
        k = int(L / 2 - w / 2)
        l = int(L / 2 + w / 2 - 1)
        tmp[i:j,k:l]=I
        I=tmp
    if h_0!=w_0:
        if not (args.s):
            print("Non square target intensity specified. Target intensity will be extended with zeros to be square.")
        L=max(h_0,w_0) #size of the square
        tmp=np.zeros((L,L))
        i=int(L/2-h_0/2)
        j=int(L/2+h_0/2-1)
        k = int(L / 2 - w_0 / 2)
        l = int(L / 2 + w_0 / 2 - 1)
        tmp[i:j,k:l]=I0
        I0=tmp

    #Conversion of the initial phase to rad
    if args.phi0:
        phi0 = ((SLM_levels/2)*np.ones(phi0.shape)-phi0) * (2 * np.pi / SLM_levels)
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
    # propagate the computed solution to image plane
    N=I0.shape[0]
    phi0_sr = np.ones((N,N)) #signal region
    phi0_sr[np.where(I0==0)[0], np.where(I0==0)[1]]=0
    phi0_sr[np.where(I0>0)[0], np.where(I0>0)[1]]=1
    A = Begin(size, wavelength, N)
    A = SubIntensity(I0, A)
    A = SubPhase(phi+phi0, A) #add source beam phase
    A = Forvard(z, A)
    I_final = np.reshape(Intensity(0, A), (N, N))

    #Compute RMS
    RMS_1=np.sqrt(np.mean(mask_sr*(I-I_final)**2))
    #compute correlation
    corr=False
    if corr and not(args.s):
        print("Computing phase correlation")
        T0=time()
        Corr=np.corrcoef(phi0_sr*phi0, phi0_sr*Phi)
        T1=time()-T0
        print(f"Done ! It took me {T1} s. Mean correlation is {np.mean(Corr)}")
    vmin=np.min(mask_sr*I)
    vmax=np.max(mask_sr*I)
    #compute RMS
    RMS=(1/(np.max(I)-np.min(I)))*np.sqrt(np.mean(phi0_sr*(I-I_final)**2))
    #save results
    plt.imsave(f"{results_path}/I0.png",I0, vmin=vmin, vmax=vmax, cmap='gray')
    plt.imsave(f"{results_path}/I.png",I, vmin=vmin, vmax=vmax, cmap='gray')
    plt.imsave(f"{results_path}/I_final.png",I_final, vmin=vmin, vmax=vmax, cmap='gray')
    plt.imsave(f"{results_path}/phi0.png",phi0, cmap='gray')
    plt.imsave(f"{results_path}/phi.png",phi, cmap='gray')
    f_rms=open(f"{results_path}/RMS_intensity.txt", "w+")
    f_rms.write(f"RMS for the intensity is : {RMS}")
    f_rms.close()
    # Plot results : intensity and phase
    #min and max intensities in the signal region for proper normalization
    if not(args.s):
        fig = plt.figure(0)
        if corr:
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
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
        im1=ax1.imshow(phi, cmap="gray", vmin=-np.pi, vmax=np.pi)
        ax1.set_title(f"Mean reconstructed phase")
        ax1.text(8, 18, f"RMS = {round(RMS, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
        fig.colorbar(im1, cax = cax1)
        im2=ax2.imshow(I, cmap="gray", vmin=vmin, vmax=vmax)
        ax2.set_title("Target intensity")
        fig.colorbar(im2, cax = cax2)
        im3=ax3.imshow(I_final, cmap="gray", vmin=vmin, vmax=vmax)
        ax3.text(8, 18, f"RMS = {round(RMS_1, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
        ax3.set_title("Propagated intensity (with mean recontructed phase)")
        fig.colorbar(im3, cax = cax3)
        if corr:
            im4=ax4.imshow(Corr, cmap="gray")
            ax4.set_title("Correlation between target phase and reconstructed phase")
            fig.colorbar(im4, cax = cax4)
        plt.show()
if __name__ == "__main__":
	main()
