# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
Update of the Matlab code from Yicheng WU
"""


import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
import configparser
import cupy as cp
from cupyx.scipy.ndimage import zoom
from cupyx.scipy import fftpack



"""
IMPORTANT NOTE : If the cupy module won't work, check that you have the right version of CuPy installed for you version
of CUDA Toolkit : https://docs-cupy.chainer.org/en/stable/install.html  
If you are sure of you CuPy install, then it is possible that your nvidia kernel module froze or that some program 
bars the access to CuPy. In this case reload your Nvidia module using these commands (in Unix) :
    sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm
This usually happens after waking up you computer. A CPU version of the code is also available WISH_lkb_cpu.py
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
        self.Nim = self.N_mod * self.N_os
        self.threshold = float(conf['params']['mask_threshold'])  # intensity threshold for the signal region
        self.noise = float(conf['params']['noise'])
    def define_mask(self, I: np.ndarray, plot: bool = False):
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
            k,l = i_min, i_max
        else:
            mask_sr[j_min:j_max, j_min:j_max] = 1
            k,l = j_min, j_max
        if plot:
            fig = plt.figure(0)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            im1=ax1.imshow(I, cmap="viridis")
            ax1.set_title("Intensity")
            im2=ax2.imshow(mask_sr, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max], color='r')
            scat.set_label('Threshold point')
            ax2.legend()
            fig.colorbar(im1, cax=cax1)
            fig.colorbar(im2, cax=cax2)
            plt.show()
        return mask_sr, k,l
    def crop_center(self, img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]
    @staticmethod
    def modulate(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        M = cp.random.rand(h, w)  # random matrix between [-1 , 1]
        phi_m = cp.asnumpy(zoom(M, (shape[0]/M.shape[0], shape[1]/M.shape[1])))
        return phi_m
    @staticmethod
    def modulate_binary(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        M = cp.random.choice(cp.asarray([0,1]), (h,w))   # random intensity mask
        #phi_m = np.kron(M, np.ones((10, 10)))
        phi_m = cp.asnumpy(zoom(M, shape[0]/M.shape[0]))
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
    @staticmethod
    def frt( A0: np.ndarray, d1: float, wv: float, z: float):
        """
        Implements propagation using Fresnel diffraction
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        k = 2*np.pi / wv
        N = A0.shape[0]
        x = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        y = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        d2 = wv * abs(z) / (N*d1)
        X1, Y1 = d1 * np.meshgrid(x, y)[0], d1 * np.meshgrid(x, y)[1]
        X2, Y2 = d2 * np.meshgrid(x, y)[0], d2 * np.meshgrid(x, y)[1]
        R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1 /(1j*wv*z)
        Q1 = np.exp(1j*(k/(2*z))*R1**2)
        Q2 = np.exp(1j*(k/(2*z))*R2**2)
        if z >=0:
            A = D * Q2 * (d1**2) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A0 * Q1, axes=(0,1)), axes=(0,1)), axes=(0,1))
        elif z<0:
            A = D * Q2 * ((N*d1) ** 2) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A0 * Q1, axes=(0,1)), axes=(0,1)), axes=(0,1))
        return A
    @staticmethod
    def frt_gpu(A0: np.ndarray, d1: float, wv: float, z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        k = 2*np.pi / wv
        N = A0.shape[0]
        x = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        y = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        d2 = wv * abs(z) / (N*d1)
        X1, Y1 = d1 * cp.meshgrid(x, y)[0], d1 * cp.meshgrid(x, y)[1]
        X2, Y2 = d2 * cp.meshgrid(x, y)[0], d2 * cp.meshgrid(x, y)[1]
        R1 = cp.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = cp.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1 /(1j*wv*z)
        Q1 = cp.exp(1j*(k/(2*z))*R1**2)
        Q2 = cp.exp(1j*(k/(2*z))*R2**2)
        if z >=0:
            A =D * Q2 * (d1**2) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0 * Q1, axes=(0,1)), axes=(0,1)), axes=(0,1))
        elif z<0:
            A =D * Q2 * ((N*d1) ** 2) * cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0 * Q1, axes=(0,1)), axes=(0,1)), axes=(0,1))

        return A
    @staticmethod
    def frt_gpu_vec(A0: np.ndarray, d1: float, wv: float, z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        k = 2 * np.pi / wv
        N = A0.shape[1]
        x = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        y = cp.linspace(0, N - 1, N) - (N / 2) * cp.ones(N)
        d2 = wv * abs(z) / (N * d1)
        X1, Y1 = d1 * cp.meshgrid(x, y)[0], d1 * cp.meshgrid(x, y)[1]
        X2, Y2 = d2 * cp.meshgrid(x, y)[0], d2 * cp.meshgrid(x, y)[1]
        R1 = cp.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = cp.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1 / (1j * wv * z)
        Q1 = cp.exp(1j * (k / (2 * z)) * R1 ** 2)
        Q2 = cp.exp(1j * (k / (2 * z)) * R2 ** 2)
        if z >= 0:
            A = D * Q2 * (d1 ** 2) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0 * Q1, axes=(1, 2)), axes=(1, 2)),
                                                     axes=(1, 2))
        elif z < 0:
            A = D * Q2 * ((N * d1) ** 2) * cp.fft.fftshift(
                cp.fft.ifft2(cp.fft.ifftshift(A0 * Q1, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

        return A
    @staticmethod
    def frt_gpu_s(A0: np.ndarray, d1: float, wv: float, z: float):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        N = A0.shape[0]
        D = 1 /(1j*wv*z)
        if z >=0:
            A =cp.multiply(D*(d1**2), cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0, axes=(0,1)), axes=(0,1)), axes=(0,1)))
        elif z<0:
            A =cp.multiply(D * ((N*d1) ** 2), cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0, axes=(0,1)), axes=(0,1)), axes=(0,1)))

        return A
    @staticmethod
    def frt_gpu_vec_s(A0: np.ndarray, d1: float, wv: float, z: float, plan=None):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1: Sampling size of the field A0
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        N = A0.shape[1]
        D = 1 / (1j * wv * abs(z))
        if z >= 0:
            if plan==None:
                A = cp.multiply(D * (d1 ** 2),
                            cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)))
            else :
                with plan:
                    A = cp.multiply(D * (d1 ** 2),
                                cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0, axes=(1, 2)), axes=(1, 2)),
                                                axes=(1, 2)))
        elif z < 0:
            if plan==None:
                A = cp.multiply(D * ((N * d1) ** 2),
                            cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)))
            else :
                with plan:
                    A = cp.multiply(D * ((N * d1) ** 2),
                                cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0, axes=(1, 2)), axes=(1, 2)),
                                                axes=(1, 2)))

        return A
    def u4Tou3(self, u4: np.ndarray, delta4: float, z3: float):
        """
        Propagates back a field from the sensor plane to the SLM plane
        :param u4: Field to propagate back
        :param delta4: Sampling size of the field u4
        :param z3: Propagation distance in metres
        :return: u3 the back propagated field
        """
        u3 = self.frt(u4, delta4, self.wavelength, -z3);
        return u3
    def process_SLM(self, slm: np.ndarray, N: int, delta3: float, type: str):
        """
        Scales the pre submitted SLM plane field (either amplitude of phase) to the right size taking into account the
        apparent size of the SLM in the sensor field of view.
        :param slm: Input SLM patterns
        :param N: Size of the calculation (typically the sensor number of pixels)
        :param delta3: Sampling size of the SLM plane (typically the "apparent" sampling size wvl*z/N*d_Sensor )
        :param type : "amp" / "phi" amplitude or phase pattern.
        :return SLM: Rescaled and properly shaped SLM patterns of size (N,N,N_batch)
        """
        delta_SLM = self.d_SLM
        N_batch = self.N_mod
        if slm.dtype == 'uint8':
            slm = slm.astype(float)/256.
        if slm.ndim == 3:
            #slm2 = slm[:, 421 : 1501, 0:N_batch] #takes a 1080x1080 square of the SLM
            slm2 = slm[:, :, 0:N_batch]
            slm3 = np.empty((N,N,N_batch))
            #scale SLM slices to the right size
            for i in range(N_batch):
                slm1 = cp.asnumpy(zoom(cp.asarray(slm2[:,:,i]), delta_SLM / delta3))
                if slm1.shape[0]>N or slm1.shape[1]>N:
                    #print("\rWARNING : The propagation distance must be too small and the field on the sensor is cropped !")
                    slm3[:,:,i]=self.crop_center(slm1, N, N)
                else :
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
                SLM = slm3.astype(np.complex64)
            else :
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        elif slm.ndim == 2:
            #slm2 = slm[:, 421:1501]
            slm2 = slm
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
                SLM = slm3.astype(np.complex64)
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        return SLM
    def gen_ims(self, u3: np.ndarray, slm: np.ndarray, z3: float, delta3: float, noise: float):
        """
        Generates dummy signal in the sensor plane from the pre generated SLM patterns
        :param u3: Initial field in the SLM plane
        :param phi0 : Initial phase typically the calibration of the SLM
        :param slm : Pre generated slm patterns
        :param z3: Propagation distance in metres
        :param delta3: "apparent" sampling size of the SLM plane (as seen by the image plane from z3 m away)
        :param noise: Intensity of the gaussian noise added to the images
        :return ims: Generated signal in the sensor plane of size (N,N,Nim)
        """
        N = u3.shape[0]
        Nim = self.Nim
        N_batch = self.N_mod
        N_os = self.N_os
        delta_SLM = self.d_SLM
        L_SLM = delta_SLM * 1080
        x = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        y = np.linspace(0, N - 1, N) - (N / 2) * np.ones(N)
        XX, YY = np.meshgrid(x,y)
        A_SLM = (np.abs(XX) * delta3 < L_SLM / 2) * (np.abs(YY) * delta3 < L_SLM / 2)

        if slm.dtype=='uint8':
            slm = slm.astype(float)/256
        ims = np.zeros((N, N, Nim), dtype=np.float32)
        for i in range(Nim):
            sys.stdout.write(f"\rGenerating image {i+1} out of {Nim} ...")
            sys.stdout.flush()
            a31 = u3 * A_SLM * slm[:,:,i//N_os]
            a31 = cp.asarray(a31)  #put the field in the GPU
            a4 = self.frt_gpu(a31, delta3, self.wavelength, z3)
            w = noise * cp.random.standard_normal((N, N,), dtype=float)
            ya = cp.abs(a4)**2 + w
            ya[ya<0]=0
            ims[:,:, i] = cp.asnumpy(ya)
            del a31, a4, ya
        return ims
    def process_ims(self, ims: np.ndarray, N: int):
        """
        Converts images to amplitudes and eventually resizes them.
        :param ims: images to convert
        :param N: Size of the sensor
        :return y0 : Processed field of size (N,N, Nim)
        """
        if ims.dtype=='uint8':
            ims=(ims/256).astype(np.float32)
        y0 = np.real(np.sqrt(ims)); # change from intensity to magnitude
        y0 = np.pad(y0, (round((N - y0.shape[0]) / 2), round((N - y0.shape[1]) / 2)))
        if y0.shape[0] > N:
            y0=y0[0:N,0:N,:]
        return y0.astype(np.float32)
    def WISHrun(self, y0: np.ndarray, SLM: np.ndarray, delta3: float, delta4: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3: Apparent sampling size of the SLM as seen from the sensor plane
        :param delta4: Sampling size of the sensor plane
        :return u3_est, u4_est, idx_converge: Estimated fields of size (N,N) and the convergence indices to check
        convergence speed
        """
        wvl = self.wavelength
        z3 = self.z
        ## parameters
        N = y0.shape[0]
        N_batch = self.N_mod
        N_os = self.N_os
        N_iter = self.N_gs
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
        del xx, yy, X, Y, R
        SLM_batch = cp.asarray(SLM[:, :, 0])
        for ii in range(N_os):
            y0_batch = cp.asarray(y0[:,:, ii])
            u3_batch[:,:, ii] = self.frt_gpu_s(y0_batch/Q, delta4, self.wavelength, -z3) * cp.conj(SLM_batch) #y0_batch gpu
        u3 = cp.mean(u3_batch, 2)
        #i_mask, j_mask = self.define_mask(np.abs(y0[:, :, 0]) ** 2, plot=True)[1:3]
        ## Recon run : GS loop
        idx_converge = np.empty(N_iter)
        for jj in range(N_iter):
            sys.stdout.flush()
            u3_collect = cp.zeros(u3.shape, dtype=cp.complex64)
            idx_converge0 = np.empty(N_batch)
            for idx_batch in range(N_batch):
                # put the correct batch into the GPU
                SLM_batch = SLM[:,:, idx_batch]
                y0_batch = y0[:,:, int(N_os * idx_batch): int(N_os * (idx_batch+1))]
                y0_batch = cp.asarray(y0_batch)
                SLM_batch=cp.asarray(SLM_batch)
                for _ in range(N_os):
                    u4[:,:,_] = self.frt_gpu_s(u3 * SLM_batch, delta3, self.wavelength, z3) # U4 is the field on the sensor
                    y[:,:,_] = y0_batch[:,:,_] * cp.exp(1j * cp.angle(u4[:,:,_])) #impose the amplitude
                    #[:,:,_] = u4[:,:,_]
                    #y[i_mask:j_mask,i_mask:j_mask,_] = y0_batch[i_mask:j_mask,i_mask:j_mask,_] \
                    #                                   * cp.exp(1j * cp.angle(u4[i_mask:j_mask,i_mask:j_mask,_]))
                    u3_batch[:,:,_] = self.frt_gpu_s(y[:,:,_], delta4, self.wavelength, -z3) * cp.conj(SLM_batch)
                u3_collect = u3_collect + cp.mean(u3_batch, 2) # collect(add) U3 from each batch
                # convergence index matrix for each batch
                idx_converge0[idx_batch] = (1/N)*\
                                           cp.linalg.norm((cp.abs(u4)-(1/N**2)*cp.sum(cp.abs(SLM_batch))*
                                                                 y0_batch)*(y0_batch>0)) #eventual mask absorption

            u3 = (u3_collect / N_batch) # average over batches
            idx_converge[jj] = np.mean(idx_converge0) # sum over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")

            # exit if the matrix doesn 't change much
            if jj > 1:
                if cp.abs(idx_converge[jj] - idx_converge[jj - 1]) / idx_converge[jj] < 5e-5:
                #if cp.abs(idx_converge[jj]) < 5e-3:
                #if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    #idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj])
                    break
        u4_est = self.frt_gpu_s(u3, delta3, self.wavelength, z3) * Q #propagate solution to sensor plane
        return u3, u4_est, idx_converge
    def WISHrun_vec(self, y0: np.ndarray, SLM: np.ndarray, delta3: float, delta4: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3: Apparent sampling size of the SLM as seen from the sensor plane
        :param delta4: Sampling size of the sensor plane
        :return u3_est, u4_est, idx_converge: Estimated fields of size (N,N) and the convergence indices to check
        convergence speed
        """
        wvl = self.wavelength
        z3 = self.z
        ## parameters
        N = y0.shape[0]
        Nim = self.Nim
        N_os = self.N_os
        N_iter = self.N_gs
        U3 = cp.empty((Nim, N, N), dtype=cp.complex64)  # store all U3 gpu
        ## initilize a3
        k = 2 * np.pi / wvl
        xx = cp.linspace(0, N - 1, N, dtype=cp.float) - (N / 2) * cp.ones(N, dtype=cp.float)
        yy = cp.linspace(0, N - 1, N, dtype=cp.float) - (N / 2) * cp.ones(N, dtype=cp.float)
        X, Y = float(delta4) * cp.meshgrid(xx, yy)[0], float(delta4) * cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j * (k / (2 * z3)) * R ** 2)
        del xx, yy, X, Y, R
        SLM = cp.asarray(SLM.repeat(N_os, axis=2))
        y0 = cp.asarray(y0)
        SLM=SLM.transpose(2,0,1)
        y0=y0.transpose(2,0,1)
        for ii in range(N_os):
            y0_batch = y0[ii, :, :]
            SLM_batch = y0[ii, :, :]
            U3[ii, :, :] = self.frt_gpu_s(y0_batch / Q, delta4, self.wavelength, -z3) * cp.conj(
                SLM_batch)  # y0_batch gpu
        u3 = cp.mean(U3[0:N_os, :, :], 0)
        del SLM_batch, y0_batch
        ## Recon run : GS loop
        idx_converge = np.empty(N_iter)
        plan_fft = fftpack.get_fft_plan(U3, axes=(1,2))
        for jj in range(N_iter):
            sys.stdout.flush()
            U3 = self.frt_gpu_vec_s((SLM * u3), delta3, self.wavelength,z3, plan=plan_fft)  #on the sensor
            # convergence index matrix for each batch
            idx_converge0 = (1 / N) * cp.linalg.norm((cp.abs(U3) -y0) * (y0 > 0), axis=(1, 2))
            U3 = y0 * cp.exp(1j * cp.angle(U3))  # impose the amplitude
            U3 = self.frt_gpu_vec_s(U3, delta4, self.wavelength, -z3, plan=plan_fft) * cp.conj(SLM)

            u3 = cp.mean(U3, 0)  # average over batches
            idx_converge[jj] = np.mean(idx_converge0)  # sum over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")

            # exit if the matrix doesn't change much
            if jj > 1:
                if cp.abs(idx_converge[jj] - idx_converge[jj - 1]) / idx_converge[jj] < 5e-5:
                #if cp.abs(idx_converge[jj]) < 5e-6:
                    # if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    # idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj])
                    break
        u4_est = self.frt_gpu_s(u3, delta3, self.wavelength, z3) * Q  # propagate solution to sensor plane
        return u3, u4_est, idx_converge


