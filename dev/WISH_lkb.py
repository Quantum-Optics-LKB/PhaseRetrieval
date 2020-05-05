# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
After the Matlab code from Yicheng WU
"""


import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
import configparser
from scipy import io
import cupy as cp
import cupyx
from scipy.ndimage import  zoom, gaussian_filter

"""
IMPORTANT NOTE : If the cupy module won't work, check that you have the right version of CuPy installed for you version
of CUDA Toolkit : https://docs-cupy.chainer.org/en/stable/install.html  
If you are sure of you CuPy install, then it is possible that your nvidia kernel module froze or that some program 
bars the access to CuPy. In this case reload your Nvidia module using these commands (in Unix) :
    sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm
This usually happens after waking up you computer. You can always remove the lines with cupy code / "gpu" functions and 
replace them with  the surrounding commented lines to run the code in CPU mode.
"""




class WISH_Sensor:
    def __init__(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path)
        self.d_SLM = float(conf["params"]["d_SLM"])
        self.d_CAM = float(conf["params"]["d_CAM"])
        self.wavelength = float(conf["params"]["wavelength"])
        self.z = float(conf["params"]["z"])  # propagation distance
        self.N_gs = int(conf["params"]["N_gs"])  # number of GS iterations
        self.N_mod = int(conf["params"]["N_mod"])  # number of modulation steps
        self.N_os = int(conf["params"]["N_os"])   #number of observations per image (to avg noise)
        self.mod_intensity = float(conf["params"]["mod_intensity"])  # modulation intensity
        self.threshold = float(conf['params']['mask_threshold'])  # intensity threshold for the signal region
        self.noise = float(conf['params']['noise'])
    def define_mask(self, I: np.ndarray, plot: bool):
        """
        A function to define the signal region automatically from the provided intensity and threshold
        :param I: intensity from which to define a signal region
        :param threshold: intensities below threshold are discarded
        :param plot: Plot or not the defined mask
        :return: mask_sr the defined mask
        """
        threshold = self.threshold
        h, w = I.shape
        mask_sr = np.zeros((h, w))
        # detect outermost non zero target intensity point
        non_zero = np.array(np.where(I > self.threshold))
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
            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)
            divider3 = make_axes_locatable(ax3)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            im1=ax1.imshow(I, cmap="viridis")
            ax1.set_title("Intensity")
            im2=ax2.imshow(mask_sr, cmap="viridis")
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max], color='r')
            scat.set_label('Threshold point')
            ax2.legend()
            extent = [min(freq), max(freq), min(freq), max(freq)]
            im3 = ax3.imshow(I_tf, cmap="viridis", extent=extent)
            ax3.set_title("Fourier transform")
            fig.colorbar(im1, cax=cax1)
            fig.colorbar(im2, cax=cax2)
            fig.colorbar(im3, cax=cax3)
            plt.show()
        return mask_sr
    def modulate(self, shape: tuple):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        x = self.mod_intensity
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(shape[0] / 10), int(shape[1] / 10)
        M = np.random.rand(h, w)  # random matrix between [-1 , 1]
        #phi_m = np.kron(M, np.ones((10, 10)))
        #phi_m = gaussian_filter(phi_m, sigma=2)
        phi_m = zoom(M, shape[0]/M.shape[0])
        phi_m =  x * phi_m
        return phi_m
    def modulate_binary(self, shape: tuple):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(shape[0] / 1), int(shape[1] / 1)
        M = np.random.choice(np.array([0,1]), (h,w), p=[0.5,0.5])   # random intensity mask
        #phi_m = np.kron(M, np.ones((10, 10)))
        #phi_m = gaussian_filter(phi_m, sigma=2)
        phi_m = zoom(M, shape[0]/M.shape[0])
        return phi_m
    def gaussian_profile(self, I: np.ndarray, sigma: float):
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
    def frt(self, A0: np.ndarray, d1: float, z: float):
        """
        Implements propagation using Fresnel diffraction
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        wv = self.wavelength
        k = 2*np.pi / wv
        N = A0.shape[0]
        x = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        y = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        d2 = wv * z / (N*d1)
        X1, Y1 = d1 * np.meshgrid(x, y)[0], d1 * np.meshgrid(x, y)[1]
        X2, Y2 = d2 * np.meshgrid(x, y)[0], d2 * np.meshgrid(x, y)[1]
        R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1 /(1j*wv*abs(z))
        Q1 = np.exp(1j*(k/(2*z))*R1**2)
        Q2 = np.exp(1j*(k/(2*z))*R2**2)
        if z >=0:
            A = D * Q2 * (d1**2) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A0 * Q1), norm='ortho'))
            #A = D * (d1**2) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A0 ), norm='ortho'))
        elif z<0:
            A = D * Q2 * ((N*d1) ** 2) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A0 * Q1), norm='ortho'))
            #A = D * ((N*d1) ** 2) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A0 ), norm='ortho'))
        #A = A/np.max(np.abs(A))
        return A
    def frt_s(self, A0: np.ndarray, d1: float, z: float):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        wv = self.wavelength
        k = 2*np.pi / wv
        N = A0.shape[0]
        D = 1 /(1j*wv*abs(z))
        if z >=0:
            A =D * (d1**2) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A0), norm='ortho'))
        elif z<0:
            A =D * ((N*d1) ** 2) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A0), norm='ortho'))

        return A
    def frt_gpu(self, A0: np.ndarray, d1: float, z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        wv = self.wavelength
        k = 2*np.pi / wv
        N = A0.shape[0]
        x = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        y = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        d2 = wv * z / (N*d1)
        X1, Y1 = d1 * cp.meshgrid(x, y)[0], d1 * cp.meshgrid(x, y)[1]
        X2, Y2 = d2 * cp.meshgrid(x, y)[0], d2 * cp.meshgrid(x, y)[1]
        R1 = cp.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = cp.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1 /(1j*wv*abs(z))
        Q1 = cp.exp(1j*(k/(2*z))*R1**2)
        Q2 = cp.exp(1j*(k/(2*z))*R2**2)
        if z >=0:
            A =D * Q2 * (d1**2) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0 * Q1), norm='ortho'))
        elif z<0:
            A =D * Q2 * ((N*d1) ** 2) * cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0 * Q1), norm='ortho'))

        return A
    def frt_gpu_s(self, A0: np.ndarray, d1: float, z: float):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        wv = self.wavelength
        k = 2*np.pi / wv
        N = A0.shape[0]
        D = 1 /(1j*wv*abs(z))
        if z >=0:
            A =D * (d1**2) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0), norm='ortho'))
        elif z<0:
            A =D * ((N*d1) ** 2) * cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0), norm='ortho'))

        return A
    def u4Tou3(self, u4: np.ndarray, delta4: float, z3: float):
        """
        Propagates back a field from the sensor plane to the SLM plane
        :param u4: Field to propagate back
        :param delta4: Sampling size of the field u4
        :param z3: Propagation distance in metres
        :return: u3 the back propagated field
        """
        u3 = self.frt(u4, delta4, -z3);
        return u3
    def gen_ims(self, u3: np.ndarray, slm: np.ndarray, z3: float, delta3: float, Nim: int, noise: float):
        """
        Generates dummy signal in the sensor plane from the pre generated SLM patterns
        :param u3: Initial field in the SLM plane
        :param phi0 : Initial phase typically the calibration of the SLM
        :param slm : Pre generated slm patterns
        :param z3: Propagation distance in metres
        :param delta3: "apparent" sampling size of the SLM plane (as seen by the image plane from z3 m away)
        :param Nim: Number of images to generate
        :param noise: Intensity of the gaussian noise added to the images
        :return ims: Generated signal in the sensor plane of size (N,N,Nim)
        """
        if Nim > 60:
            print('max Nim is 60.')
            raise
        N = u3.shape[0]
        delta_SLM = self.d_SLM
        L_SLM = delta_SLM * 1080
        x = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        y = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        XX, YY = np.meshgrid(x,y)
        A_SLM = (np.abs(XX) * delta3 < L_SLM / 2) * (np.abs(YY) * delta3 < L_SLM / 2)

        if slm.dtype=='uint8':
            slm = slm.astype(float)/256
        ims = np.zeros((N, N, Nim), dtype=float)
        for i in range(Nim):
            sys.stdout.write(f"\rGenerating image {i+1} out of {Nim} ...")
            sys.stdout.flush()
            #a31 = u3 * A_SLM * np.exp(1j * slm[:,:,i] * 2 * np.pi)
            a31 = u3 * A_SLM * slm[:,:,i]
            a31 = cp.asarray(a31)  #put the field in the GPU
            #a4 = self.frt(a31, delta3, z3)
            a4 = self.frt_gpu(a31, delta3, z3)
            w = noise * cp.random.rand(N, N)
            ya = cp.abs(a4)**2 + w
            ya[ya<0]=0
            #ims[:,:, i] = ya
            ims[:,:, i] = cp.asnumpy(ya)

        return ims
    def process_SLM(self, slm: np.ndarray, N: int, Nim: int, delta3: float, type: str):
        """
        Scales the pre submitted SLM plane field (either amplitude of phase) to the right size taking into account the
        apparent size of the SLM in the sensor field of view.
        :param slm: Input SLM patterns
        :param N: Size of the calculation (typically the sensor number of pixels)
        :param Nim: Number of images to generate
        :param delta3: Sampling size of the SLM plane (typically the "apparent" sampling size wvl*z/N*d_Sensor )
        :param type : "amp" / "phi" amplitude or phase pattern.
        :return SLM: Rescaled and properly shaped SLM patterns of size (N,N,Nim)
        """
        delta_SLM = self.d_SLM
        if slm.dtype == 'uint8':
            slm = slm.astype(float)/256.
        if slm.ndim == 3:
            slm2 = slm[:, 421: 1501, 0:Nim] #takes a 1080x1080 square of the SLM
            #slm2 = slm[:, :, 0:Nim] #takes a 1080x1080 square of the SLM
            slm3 = np.empty((N,N,Nim))
            #could replace with my modulate function
            #scale SLM slices to the right size
            for i in range(Nim):
                slm1 = zoom(slm2[:,:,i], delta_SLM / delta3)
                if slm1.shape[0]>N or slm1.shape[1]>N:
                    print("ERROR : The propagation distance must be too small and the field on the sensor is cropped !")
                slm1 = np.pad(slm1, (int(np.ceil((N - slm1.shape[0]) / 2)), \
                                     int(np.ceil((N - slm1.shape[1]) / 2))))
                if slm1.shape[0] > N and slm1.shape[1] > N:
                    slm3[:, :, i] = slm1[0:N, 0:N]
                elif slm1.shape[0] > N:
                    slm3[:,:,i] = slm1[0:N, :]
                elif slm1.shape[1] > N:
                    slm3[:,:,i] = slm1[:, 0:N]
                else :
                    slm3[:,:,i] = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3
            else :
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        elif slm.ndim == 2:
            slm2 = slm[:, 421:1501]
            #slm2 = slm
            slm3 = np.empty((N, N))
            # could replace with my modulate function
            # scale SLM slices to the right size
            slm1 = zoom(slm2, delta_SLM / delta3)
            slm1 = np.pad(slm1, (int(np.ceil((N - slm1.shape[0]) / 2)), \
                                 int(np.ceil((N - slm1.shape[1]) / 2))))
            if slm1.shape[0] > N and slm1.shape[1] > N:
                slm3 = slm1[0:N, 0:N]
            elif slm1.shape[0] > N:
                slm3 = slm1[0:N, :]
            elif slm1.shape[1] > N:
                slm3 = slm1[:, 0:N]
            else:
                slm3 = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        return SLM

    def process_CAM(self, slm: np.ndarray, N: int, Nim: int, delta3: float, type: str):
        """
        Scales the pre submitted SLM plane field (either amplitude of phase) to the right size taking into account the
        apparent size of the SLM in the sensor field of view.
        :param slm: Input SLM patterns
        :param N: Size of the calculation (typically the sensor number of pixels)
        :param Nim: Number of images to generate
        :param delta3: Sampling size of the SLM plane (typically the "apparent" sampling size wvl*z/N*d_Sensor )
        :param type : "amp" / "phi" amplitude or phase pattern.
        :return SLM: Rescaled and properly shaped SLM patterns of size (N,N,Nim)
        """
        delta_CAM = self.d_CAM
        if slm.dtype == 'uint8':
            slm = slm.astype(float) / 256
        # slm2 = slm[:, 421: 1501, 0:Nim] #takes a 1080x1080 square of the SLM
        if slm.ndim == 3:
            slm2 = slm[:, :, 0:Nim]
            slm3 = np.empty((N, N, Nim))
            # could replace with my modulate function
            # scale SLM slices to the right size
            for i in range(Nim):
                slm1 = zoom(slm2[:, :, i], delta_CAM / delta3)
                slm1 = np.pad(slm1, (int(np.ceil((N - slm1.shape[0]) / 2)), \
                                     int(np.ceil((N - slm1.shape[1]) / 2))))
                if slm1.shape[0] > N and slm1.shape[1] > N:
                    slm3[:, :, i] = slm1[0:N, 0:N]
                elif slm1.shape[0] > N:
                    slm3[:, :, i] = slm1[0:N, :]
                elif slm1.shape[1] > N:
                    slm3[:, :, i] = slm1[:, 0:N]
                else:
                    slm3[:, :, i] = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        elif slm.ndim == 2:
            slm2 = slm
            slm3 = np.empty((N, N))
            # could replace with my modulate function
            # scale SLM slices to the right size
            slm1 = zoom(slm2, delta_CAM / delta3)
            slm1 = np.pad(slm1, (int(np.ceil((N - slm1.shape[0]) / 2)), \
                                 int(np.ceil((N - slm1.shape[1]) / 2))))
            if slm1.shape[0] > N and slm1.shape[1] > N:
                slm3 = slm1[0:N, 0:N]
            elif slm1.shape[0] > N:
                slm3 = slm1[0:N, :]
            elif slm1.shape[1] > N:
                slm3 = slm1[:, 0:N]
            else:
                slm3 = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        return SLM
    def process_ims(self, ims: np.ndarray, N: int):
        """
        Converts images to amplitudes and eventually resizes them.
        :param ims: images to convert
        :param N: Size of the sensor
        :return y0 : Processed field of size (N,N, Nim)
        """
        y0 = np.real(np.sqrt(ims)); # change from intensity to magnitude
        y0 = np.pad(y0, (round((N - y0.shape[0]) / 2), round((N - y0.shape[1]) / 2)))
        if y0.shape[0] > N:
            y0=y0[0:N,0:N,:]
        return y0
    def WISHrun(self, y0: np.ndarray, SLM: np.ndarray, delta3: float, delta4: float, N_os: int, N_iter: int,\
                N_batch: int, plot: bool=True):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3: Apparent sampling size of the SLM as seen from the sensor plane
        :param delta4: Sampling size of the sensor plane
        :param N_os: Number of observations per image
        :param N_iter: Maximal number of Gerchberg Saxton iterations
        :param N_batch: Number of batches (modulations)
        :param plot: If True, plots the advance of the retrieval every 10 iterations
        :return u4_est, idx_converge: Estimated field of size (N,N) and the convergence indices to check convergence
                                      speed
        """
        wvl = self.wavelength
        z3 = self.z
        ## parameters
        N = y0.shape[0]
        #u3_batch = np.zeros((N, N, N_os), dtype=complex) # store all U3 gpu
        #u4 = np.zeros((N, N, N_os), dtype=complex) # gpu
        #y = np.zeros((N, N, N_os), dtype=complex) # store all U3 gpu
        u3_batch = cp.zeros((N, N, N_os), dtype=cp.complex64) # store all U3 gpu
        u4 = cp.zeros((N, N, N_os), dtype=cp.complex64) # gpu
        y = cp.zeros((N, N, N_os), dtype=cp.complex64) # store all U3 gpu

        ## initilize a3
        k = 2 * np.pi / wvl
        xx = cp.linspace(0, N - 1, N, dtype=cp.float) - (N / 2) * cp.ones(N, dtype=cp.float)
        yy = cp.linspace(0, N - 1, N, dtype=cp.float) - (N / 2) * cp.ones(N, dtype=cp.float)
        X, Y = float(delta4) * cp.meshgrid(xx, yy)[0], float(delta4) * cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j*(k/(2*z3))*R**2)
        for ii in range(N_os):
            #SLM_batch = SLM[:,:, ii]
            SLM_batch = cp.asarray(SLM[:,:, ii])
            y0_batch = y0[:,:, ii]
            #u3_batch[:,:, ii] = self.frt(y0_batch, delta4, -z3) * np.conj(SLM_batch) #y0_batch gpu
            #u3_batch[:,:, ii] = self.frt_gpu(cp.asarray(y0_batch), delta4, -z3) * cp.conj(SLM_batch) #y0_batch gpu
            u3_batch[:,:, ii] = self.frt_gpu_s(cp.asarray(y0_batch)/Q, delta4, -z3) * cp.conj(SLM_batch) #y0_batch gpu
        #u3 = np.mean(u3_batch, 2) # average it
        u3 = cp.mean(u3_batch, 2)

        ## Recon run : GS loop
        idx_converge = np.empty(N_iter)
        for jj in range(N_iter):
            sys.stdout.write(f"\rGS iteration {jj+1}")
            sys.stdout.flush()
            #u3_collect = np.zeros(u3.shape, dtype=complex)
            u3_collect = cp.zeros(u3.shape, dtype=cp.complex64)
            idx_converge0 = np.empty(N_batch)
            for idx_batch in range(N_batch):
                # put the correct batch into the GPU (no GPU for now)
                #SLM_batch = SLM[:,:, int(N_os * idx_batch): int(N_os * (idx_batch+1))]
                #y0_batch = y0[:,:, int(N_os * idx_batch): int(N_os * (idx_batch+1))]
                SLM_batch = cp.asarray(SLM[:,:, int(N_os * idx_batch): int(N_os * (idx_batch+1))])
                y0_batch = cp.asarray(y0[:,:, int(N_os * idx_batch): int(N_os * (idx_batch+1))])
                for _ in range(N_os):
                    #u4[:,:,_] = self.frt(u3 * SLM_batch[:,:,_], delta3, z3) # U4 is the field on the sensor
                    u4[:,:,_] = self.frt_gpu_s(u3 * SLM_batch[:,:,_], delta3, z3) # U4 is the field on the sensor
                    y[:,:,_] = y0_batch[:,:,_] * cp.exp(1j * cp.angle(u4[:,:,_])) # force the amplitude of y to be y0
                    #u3_batch[:,:,_] = self.frt(y[:,:,_], delta4, -z3) * np.conj(SLM_batch[:,:,_])
                    u3_batch[:,:,_] = self.frt_gpu_s(y[:,:,_], delta4, -z3) * cp.conj(SLM_batch[:,:,_])
                #u3_collect = u3_collect + np.mean(u3_batch, 2) # collect(add) U3 from each batch
                u3_collect = u3_collect + cp.mean(u3_batch, 2) # collect(add) U3 from each batch
                #idx_converge0[idx_batch] = np.mean(np.mean(np.mean(y0_batch,1),0)/np.sum(np.sum(np.abs(np.abs(u4)-y0_batch),1),0))
                #idx_converge0[idx_batch] = cp.asnumpy(cp.mean(cp.mean(cp.mean(y0_batch,1),0)/cp.sum(cp.sum(cp.abs(cp.abs(u4)-y0_batch),1),0)))
                # convergence index matrix for each batch
                idx_converge0[idx_batch] = (1/N)*cp.linalg.norm((cp.abs(u4)-y0_batch)*(y0_batch>0))

            u3 = (u3_collect / N_batch) # average over batches
            idx_converge[jj] = np.mean(idx_converge0) # sum over batches
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")



            if jj % 10 == 0 and plot:
                u4_est = cp.asnumpy(self.frt_gpu_s(u3, delta3, z3) * Q)
                plt.close('all')
                fig=plt.figure(0)
                fig.suptitle(f'Iteration {jj}')
                ax1=fig.add_subplot(121)
                ax2=fig.add_subplot(122)
                im=ax1.imshow(np.abs(u4_est), cmap='viridis')
                ax1.set_title('Amplitude')
                ax2.imshow(np.angle(u4_est), cmap='viridis')
                ax2.set_title('Phase')

                fig1=plt.figure(1)
                ax = fig1.gca()
                ax.plot(np.arange(0,jj,1), idx_converge[0:jj], marker='o')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Convergence estimator')
                ax.set_title('Convergence curve')
                plt.show()
                time.sleep(2)

            # exit if the matrix doesn 't change much
            if jj > 1:
                if cp.abs(idx_converge[jj] - idx_converge[jj - 1]) / idx_converge[jj] < 1e-4:
                #if cp.abs(idx_converge[jj]) < 1e-3:
                    print('\nConverged. Exit the GS loop ...')
                    #idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj])
                    break
        # u4_est = self.frt(u3, delta3, z3)
        u4_est = cp.asnumpy(self.frt_gpu_s(u3, delta3, z3) * Q) #propagate solution to sensor plane
        u3 = cp.asnumpy(u3)
        return u3, u4_est, idx_converge



#WISH routine
def main():
    #start timer
    T0 = time.time()
    #instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    #I0 = np.array(Image.open('intensities/harambe_256_full.bmp'))[:,:,0]
    #I0 = I0.astype(float)/256
    #I0 = np.pad(I0.astype(np.float) / 256, (256, 256))  # protection band
    im = np.array(Image.open('intensities/harambe_512_full.bmp'))[:,:,0]
    phi0 = np.array(Image.open('phases/calib_512_full.bmp'))
    u40 = np.pad(im.astype(np.float)/256, (512,512)) #protection band

    phi0 = np.pad(phi0.astype(np.float)/256, (512,512)) #protection band
    u40 = u40 * (np.exp(1j * phi0 * 2 * np.pi))
    N = u40.shape[0]
    #N = int((Sensor.d_SLM/Sensor.d_CAM)*I0.shape[0]+512)
    delta3 = wvl * z3 / (N * delta4)
    u30 = Sensor.u4Tou3(u40, delta4, z3)
    #u30 = np.sqrt(I0)
    #u30 = Sensor.process_CAM(u30, N, 1, delta3, type="amp")
    #PHI0 = Sensor.process_CAM(phi0, N, 1, delta3, type="phi")
    ## forward prop to the sensor plane with SLM modulation
    print('Generating simulation data images ...')
    noise = Sensor.noise
    Nim = Sensor.N_mod*Sensor.N_os
    slm = np.zeros((1080, 1920,Nim))
    q = np.zeros((1080, 1920,Nim))
    k = 2 * np.pi / wvl
    f = 4*z3
    Ny, Nx = slm[:, :, 0].shape
    xx = np.linspace(0, Nx - 1, Nx, dtype=np.float) - (Nx / 2) * np.ones(Nx, dtype=np.float)
    yy = np.linspace(0, Ny - 1, Ny, dtype=np.float) - (Ny / 2) * np.ones(Ny, dtype=np.float)
    X, Y = float(delta4) * np.meshgrid(xx, yy)[0], float(delta4) * np.meshgrid(xx, yy)[1]
    R = np.sqrt(X ** 2 + Y ** 2)
    #slm = np.zeros((1080, 1920,Nim))
    for i in range(Nim-1):
        #slm[:,:,i]=Sensor.modulate((1080,1920))
        #slm[:,:,i]=Sensor.modulate((1080,1920))
        slm[:,:,i] = Sensor.modulate_binary((1080, 1920))
        q[:,:,i] = -(1 / 2 * np.pi) * (k / (2 * f)) * R ** 2
    slm[:,:,Nim-1]=np.ones(slm[:,:,Nim-1].shape)
    q[:,:,Nim-1] = -(1 / 2 * np.pi) * (k / (2 * f)) * R ** 2
    #SLM = Sensor.process_CAM(slm, N, Nim, delta3, type="phi")
    #SLM = Sensor.process_SLM(slm, N, Nim, delta3, type="phi")
    SLM = Sensor.process_SLM(slm, N, Nim, delta3, type="amp")
    #Q = Sensor.process_SLM(q, N, Nim, delta3, type="phi")
    SLM[SLM > 0.5] = 1
    SLM[SLM <= 0.5] = 0
    #SLM=SLM.astype(np.complex64)*Q
    #u30 = Sensor.process_SLM(np.sqrt(I0), N, 1, delta3, type='amp')
    #PHI0 = Sensor.process_SLM(phi0, N, 1, delta3, type='phi')
    #u30 = u30 * PHI0
    ims = Sensor.gen_ims(u30, SLM, z3, delta3, Nim, noise)

    print('\nCaptured images are simulated')
    #clear u30
    #del u30
    #reconstruction
    #process the captured image : converting to amplitude and padding if needed
    y0 = Sensor.process_ims(ims, N)
    plt.imshow(y0[:,:,0])
    plt.show()
    ##Recon initilization
    N_os = Sensor.N_os # number of images per batch
    if Nim < N_os:
        N_os = Nim
    N_iter = Sensor.N_gs  # number of GS iterations
    N_batch = int(Nim / N_os)  # number of batches
    u3_est, u4_est, idx_converge = Sensor.WISHrun(y0, SLM, delta3, delta4, N_os, N_iter, N_batch, plot=False)
    phase_rms =(1/N)*np.linalg.norm((np.angle(u40)-np.angle(u4_est))*(np.abs(u40)>0))
    #phase_rms =(1/N)*np.linalg.norm((np.angle(u30)-np.angle(u3_est))*(np.abs(u30)>0))
    print(f"\n Phase RMS is : {phase_rms}")
    #total time
    T= time.time()-T0
    print(f"\n Total time elapsed : {T} s")

    fig=plt.figure()
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
    im1=ax1.imshow(np.abs(u40), cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Amplitude GT')
    im2=ax2.imshow(np.angle(u40), cmap='viridis',vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Phase GT')
    im3=ax3.imshow(abs(u4_est), cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Amplitude estimation')
    im4=ax4.imshow(np.angle(u4_est), cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase estimation')
    ax5.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    plt.show()
if __name__=="__main__":
    main()
