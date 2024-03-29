# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
Based on the Matlab code from Yicheng WU
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import configparser
import cupy as cp
from cupyx.scipy.ndimage import zoom as zoom_cp
from cupyx.scipy.ndimage import shift as shift_cp
from scipy.ndimage import zoom as zoom
from scipy.ndimage import shift as shift
from cupyx.scipy import fft as fftsc
from cupyx.scipy.fft import get_fft_plan
from cupyx.time import repeat as repeat_c
from timeit import repeat
import pyfftw
import mkl_fft
import multiprocessing
import time
import pickle
from ctypes import c_int

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'


"""
IMPORTANT NOTE : If the cupy module won't work, check that you have the right
version of CuPy installed for you version
of CUDA Toolkit : https://docs-cupy.chainer.org/en/stable/install.html
If you are sure of you CuPy install, then it is possible that your nvidia
kernel module froze or that some program bars the access to CuPy. In this case
reload your Nvidia module using these commands (in Unix) :
    sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm
This usually happens after waking up you computer. A CPU version of the code is
also available WISH_lkb_cpu.py
"""

# #Kernel for amplitude imposing and multplication / conjugation
# with open("kernels.cu") as file:
#     code = file.read()
#     kernels = cp.RawModule(code=code, backend="nvcc")
#     kernels.compile()
#     ker_impose_amp = kernels.get_function("impose_amp")
#     ker_multiply_conjugate = kernels.get_function("multiply_conjugate")
#     gpu_frt = kernels.get_function('frt_gpu_vec_s')
# kernels = cp.RawModule(path="kernels", options=("--use_fast_math",), backend="nvcc")
# ker_impose_amp = kernels.get_function("impose_amp")
# ker_multiply_conjugate = kernels.get_function("multiply_conjugate")
@cp.fuse()
def ker_impose_amp_norm(y, x, a):
    return cp.abs(y) * cp.exp(1j*cp.angle(a*x))

@cp.fuse()
def ker_multiply_conjugate_sum_norm(y, x, a):
    return cp.sum(a * x * cp.conj(y), axis=0)


class WISH_Sensor:
    def __init__(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path)
        self.d_SLM = float(conf["params"]["d_SLM"])
        self.d_CAM = float(conf["params"]["d_CAM"])
        self.wavelength = float(conf["params"]["wavelength"])
        # refractive index
        # self.n = float(conf["params"]["n"])
        # propagation distance
        self.z = float(conf["params"]["z"])
        # number of GS iterations
        self.N_gs = int(conf["params"]["N_gs"])
        # number of modulation steps
        self.N_mod = int(conf["params"]["N_mod"])
        # number of observations per image (to avg noise)
        self.N_os = int(conf["params"]["N_os"])
        self.Nim = self.N_mod * self.N_os
        # intensity threshold for the signal region
        self.threshold = float(conf['params']['mask_threshold'])
        self.noise = float(conf['params']['noise'])

    def define_mask(self, I: np.ndarray, plot: bool = False):
        """
        A function to define the signal region automatically from the provided
        intensity and threshold
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
            k, m = i_min, i_max
        else:
            mask_sr[j_min:j_max, j_min:j_max] = 1
            k, m = j_min, j_max
        if plot:
            fig = plt.figure(0)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            im1 = ax1.imshow(I, cmap="viridis")
            ax1.set_title("Intensity")
            im2 = ax2.imshow(mask_sr, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max],
                               color='r')
            scat.set_label('Threshold point')
            ax2.legend()
            fig.colorbar(im1, cax=cax1)
            fig.colorbar(im2, cax=cax2)
            plt.show()
        return mask_sr, k, m

    def crop_center(self, img: np.ndarray, cropx: int, cropy: int):
        """
        A function to crop around the center of an array
        :param img: Array to crop
        :param cropx: Size along x direction of cropped array
        :param cropy: Size along y direction of cropped array
        :return: cropped array
        """
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    @staticmethod
    def modulate(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too
        much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        # random matrix between [0 , 1]
        M = (1/256)*cp.random.random_integers(0, 255, (h, w))
        M = M.astype(np.float32)
        phi_m = cp.asnumpy(
                    zoom_cp(M, (shape[0]/M.shape[0], shape[1]/M.shape[1])))
        return phi_m

    @staticmethod
    def modulate_binary(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too
        much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """

        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        # random intensity mask
        M = cp.random.choice(cp.asarray([0, 0.5]), (h, w))
        phi_m = cp.asnumpy(
                zoom_cp(M, (shape[0] / M.shape[0], shape[1] / M.shape[1]),
                        order=0))
        return phi_m

    def gaussian_profile(self, I: np.ndarray, sigma: float):
        """
        Applies a gaussian profile to the intensity provided
        :param I: Intensity to which a gaussian profile is going to be applied
        :param sigma: Standard deviation of the gaussian profile, in fraction
        of the provided intensity size
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
    def frt(A0: np.ndarray, d1x: float, d1y: float, wv: float,
            z: float):
        """
        Implements propagation using Fresnel diffraction
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: index of refraction
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        k = 2*np.pi / wv
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        x = np.linspace(0, Nx - 1, Nx) - (Nx / 2) * np.ones(Nx)
        y = np.linspace(0, Ny - 1, Ny) - (Ny / 2) * np.ones(Ny)
        d2x = wv * abs(z) / (Nx*d1x)
        d2y = wv * abs(z) / (Ny*d1y)
        X1, Y1 = d1x * np.meshgrid(x, y)[0], d1y * np.meshgrid(x, y)[1]
        X2, Y2 = d2x * np.meshgrid(x, y)[0], d2y * np.meshgrid(x, y)[1]
        R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        D = 1/(1j*wv*z)
        Q1 = np.exp(1j*(k/(2*z))*R1**2)
        Q2 = np.exp(1j*(k/(2*z))*R2**2)
        if z >= 0:
            A = D * Q2 * (d1x*d1y) * np.fft.fftshift(np.fft.fft2(
                np.fft.ifftshift(A0 * Q1, axes=(0, 1)), axes=(0, 1)),
                axes=(0, 1))
        elif z < 0:
            A = D * Q2 * (Nx*d1x*Ny*d1y) * np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(
                    A0 * Q1, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        return A

    @staticmethod
    def frt_gpu(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using
        CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: Refractive index
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        k = 2 * np.pi / wv
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        x = cp.linspace(0, Nx - 1, Nx) - (Nx / 2) * cp.ones(Nx)
        y = cp.linspace(0, Ny - 1, Ny) - (Ny / 2) * cp.ones(Ny)
        d2x = wv * abs(z) / (Nx * d1x)
        d2y = wv * abs(z) / (Ny * d1y)
        X1, Y1 = d1x * cp.meshgrid(x, y)[0], d1y * cp.meshgrid(x, y)[1]
        X2, Y2 = d2x * cp.meshgrid(x, y)[0], d2y * cp.meshgrid(x, y)[1]
        R1 = cp.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = cp.sqrt(X2 ** 2 + Y2 ** 2)
        A0 = A0 * cp.exp(1j * (k / (2 * z)) * R1 ** 2)
        if z > 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = cp.fft.fft2(A0, axes=(0, 1))
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = cp.fft.ifft2(A0, axes=(0, 1))
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * cp.exp(1j * (k / (2 * z)) * R2 ** 2)
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_gpu_vec(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                    n: float, z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using
        CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: index of refraction
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        k = n*2 * np.pi / wv
        Nx = A0.shape[2]
        Ny = A0.shape[1]
        x = cp.linspace(0, Nx - 1, Nx) - (Nx / 2) * cp.ones(Nx)
        y = cp.linspace(0, Ny - 1, Ny) - (Ny / 2) * cp.ones(Ny)
        d2x = wv * abs(z) / (Nx * d1x)
        d2y = wv * abs(z) / (Ny * d1y)
        X1, Y1 = d1x * cp.meshgrid(x, y)[0], d1y * cp.meshgrid(x, y)[1]
        X2, Y2 = d2x * cp.meshgrid(x, y)[0], d2y * cp.meshgrid(x, y)[1]
        R1 = cp.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = cp.sqrt(X2 ** 2 + Y2 ** 2)
        A0 = A0 * cp.exp(1j * (k / (2 * z)) * R1 ** 2)
        if z > 0:
            A0 = cp.fft.ifftshift(A0, axes=(1, 2))
            A0 = cp.fft.fft2(A0, axes=(1, 2))
            A0 = cp.fft.fftshift(A0, axes=(1, 2))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = cp.fft.ifftshift(A0, axes=(1, 2))
            A0 = cp.fft.ifft2(A0, axes=(1, 2))
            A0 = cp.fft.fftshift(A0, axes=(1, 2))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * cp.exp(1j * (k / (2 * z)) * R2 ** 2)
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_gpu_s(A0: np.ndarray, d1x: float, d1y: float, wv: float, z: float):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a
        GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        if z > 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = fftsc.fft2(A0, axes=(0, 1), overwrite_x=True)
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 *= d1x*d1y 
        elif z <= 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = fftsc.ifft2(A0, axes=(0, 1), overwrite_x=True)
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 *= (Nx*d1x*Ny*d1y) 
        A0 *= 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_gpu_vec_s(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                      z: float, plan=None):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a
        GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        if plan is None:
            Nx = A0.shape[2]
            Ny = A0.shape[1]
        if z > 0:
            if plan is None:
                A0 = fftsc.fft2(A0, axes=(1, 2), overwrite_x=True)
                A0 *= d1x*d1y 
            else:
                plan.fft(A0, A0, cp.cuda.cufft.CUFFT_FORWARD)
                A0 *= d1x*d1y  
        elif z <= 0:
            if plan is None:
                A0 = fftsc.ifft2(A0, axes=(1, 2), overwrite_x=True)
                A0 *= (Nx*d1x*Ny*d1y) 
            else:
                plan.fft(A0, A0, cp.cuda.cufft.CUFFT_INVERSE)
                A0 *= d1x*d1y 
        A0 *= 1 / (1j * wv * z)


    def u4Tou3(self, u4: np.ndarray, delta4x: float, delta4y: float,
               z3: float):
        """
        Propagates back a field from the sensor plane to the SLM plane
        :param u4: Field to propagate back
        :param delta4x: Sampling size of the field u4 in the x direction
        :param delta4y: Sampling size of the field u4 in the y direction
        :param z3: Propagation distance in metres
        :return: u3 the back propagated field
        """
        u3 = self.frt(u4, delta4x, delta4y, self.wavelength, -z3)
        return u3

    def process_SLM(self, slm: np.ndarray, Nx: int, Ny: int, delta4x: float,
                    delta4y: float, type: str):
        """
        Scales the pre submitted SLM plane field (either amplitude of phase) to
        the right size taking into account the apparent size of the SLM in the
        sensor field of view.
        :param slm: Input SLM patterns
        :param Nx: Size of the calculation in x
        :param Ny: Size of the calculation in y
        :param delta3x: Sampling size of the SLM plane (typically the
        "apparent" sampling size wvl*z/Nx*d_Sensorx )
        :param delta3y: Sampling size of the SLM plane (typically the
        "apparent" sampling size wvl*z/Ny*d_Sensory )
        :param type : "amp" / "phi" amplitude or phase pattern.
        :return SLM: Rescaled and properly shaped SLM patterns of size
        (Ny,Nx,N_batch)
        """
        delta_SLM = self.d_SLM
        N_batch = self.N_mod
        delta3x = self.wavelength * self.z / (Nx * delta4x)
        delta3y = self.wavelength * self.z / (Ny * delta4y)
        if slm.dtype == 'uint8':
            slm = slm.astype(np.float32)/256.
        # check if SLM can be centered in the computational window
        Nxslm = slm.shape[1]
        Nyslm = slm.shape[0]
        cdx = (np.round(Nxslm*delta_SLM/delta3x) % 2) != 0
        cdy = (np.round(Nyslm*delta_SLM/delta3y) % 2) != 0
        if cdx or cdy:
            Z = np.linspace(self.z-1e-4, self.z+1e-4, int(1e3))
            D3x = self.wavelength * Z / (Nx * self.d_CAM)
            D3y = self.wavelength * Z / (Ny * self.d_CAM)
            X = np.round(Nxslm*self.d_SLM/D3x)
            Y = np.round(Nyslm*self.d_SLM/D3y)
            X = X % 2
            Y = Y % 2
            diff = X+Y
            z_corr = Z[diff == 0][0]
            print(
                "WARNING : Propagation is such that the SLM cannot be" +
                " centered in the computational window. Distance will be set" +
                " to closest matching distance" +
                f" z = {np.round(z_corr*1e3, decimals=2)} mm.")
            # print("\nPlease adjust propagation distance or continue.")
            # while cont is None:
            #     cont = input("\nContinue ? [y/n]")
            #     if cont == 'y':
            #         self.z = z_corr
            #     elif cont == 'n':
            #         exit()
            #     else:
            #         cont = None
            self.z = z_corr
        delta3x = self.wavelength * self.z / (Nx * delta4x)
        delta3y = self.wavelength * self.z / (Ny * delta4y)
        if slm.ndim == 3:
            slm3 = np.empty((Ny, Nx, N_batch))
            # scale SLM slices to the right size
            for i in range(N_batch):
                slm1 = cp.asnumpy(
                        zoom_cp(cp.asarray(slm[:, :, i]), [delta_SLM/delta3y,
                                delta_SLM/delta3x], order=0))
                if slm1.shape[0] > Ny and slm1.shape[1] <= Nx:
                    slm1 = self.crop_center(slm1, slm1.shape[1], Ny)
                elif slm1.shape[0] <= Ny and slm1.shape[1] > Nx:
                    slm1 = self.crop_center(slm1, Nx, slm1.shape[0])
                elif slm1.shape[0] > Ny and slm1.shape[1] > Nx:
                    slm1 = self.crop_center(slm1, Nx, Ny)
                if slm1.shape[0] < Ny or slm1.shape[1] < Nx:
                    pady = int(np.ceil((Ny - slm1.shape[0]) / 2))
                    padx = int(np.ceil((Nx - slm1.shape[1]) / 2))
                    slm1 = np.pad(slm1, ((pady, pady), (padx, padx)))
                slm3[:, :, i] = slm1
            if type == "phi":
                SLM = (np.exp(1j * 2 * np.pi * slm3)*(slm3 != 0)).astype(np.complex64) +\
                      (np.zeros(slm3.shape)*(slm3 == 0)).astype(np.complex64)
                # SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3.astype(np.complex64)
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        elif slm.ndim == 2:
            slm2 = slm
            slm3 = np.empty((Ny, Nx))
            # scale SLM slices to the right size
            slm1 = zoom(slm2, [delta_SLM / delta3y, delta_SLM / delta3x])
            slm1 = np.pad(slm1, (int(np.ceil((Ny - slm1.shape[0]) / 2)),
                                 int(np.ceil((Nx - slm1.shape[1]) / 2))))
            slm3 = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3.astype(np.complex64)
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        return SLM

    def gen_ims(self, u3: np.ndarray, slm: np.ndarray, z3: float,
                delta3x: float, delta3y: float, noise: float):
        """
        Generates dummy signal in the sensor plane from the pre generated SLM
        patterns
        :param u3: Initial field in the SLM plane
        :param phi0 : Initial phase typically the calibration of the SLM
        :param slm : Pre generated slm patterns
        :param z3: Propagation distance in metres
        :param delta3x: "apparent" sampling size of the SLM plane (as seen by
        the image plane from z3 m away) in the x direction
        :param delta3y: "apparent" sampling size of the SLM plane (as seen by
        the image plane from z3 m away) in the x direction
        :param noise: Intensity of the gaussian noise added to the images
        :return ims: Generated signal in the sensor plane of size (N,N,Nim)
        """
        Nx = u3.shape[1]
        Ny = u3.shape[0]
        Nim = self.Nim
        N_os = self.N_os
        delta_SLM = self.d_SLM
        Lx_SLM = delta_SLM * slm.shape[1]
        Ly_SLM = delta_SLM * slm.shape[0]
        x = np.linspace(0, Nx - 1, Nx) - (Nx / 2) * np.ones(Nx)
        y = np.linspace(0, Ny - 1, Ny) - (Ny / 2) * np.ones(Ny)
        XX, YY = np.meshgrid(x, y)
        A_SLM = (np.abs(XX) * delta3x < Lx_SLM / 2) * \
                (np.abs(YY) * delta3y < Ly_SLM / 2)

        if slm.dtype == 'uint8':
            slm = slm.astype(float)/256
        ims = np.zeros((Ny, Nx, Nim), dtype=np.float32)
        for i in range(Nim):
            sys.stdout.write(f"\rGenerating image {i+1} out of {Nim} ...")
            sys.stdout.flush()
            a31 = u3 * A_SLM * slm[:, :, i//N_os]
            a31 = cp.asarray(a31, dtype=cp.complex64)  # put the field in the GPU
            a4 = self.frt_gpu(a31, delta3x, delta3y, self.wavelength,
                              z3)
            ya = cp.abs(a4)**2
            if noise > 0.0:
                w = noise * cp.random.standard_normal((Ny, Nx,), dtype=float)
                ya += w
                ya = shift_cp(ya, (1*cp.random.standard_normal(1, dtype=float),
                              1*cp.random.standard_normal(1, dtype=float)))
                ya[ya < 0] = 0
            ims[:, :, i] = cp.asnumpy(ya)
            del a31, a4, ya
        return ims

    def process_ims(self, ims: np.ndarray, Nx: int, Ny: int):
        """
        Converts images to amplitudes and eventually resizes them.
        :param ims: images to convert
        :param Nx: Pixel nbr of the sensor along x
        :param Ny: Pixel nbr of the sensor along y
        :return y0 : Processed field of size (Nx,Ny, Nim)
        """
        if ims.dtype == 'uint8':
            ims = (ims/256).astype(np.float32)
        y0 = np.real(np.sqrt(ims))  # change from intensity to magnitude
        return y0.astype(np.float32)

    def WISHrun(self, y0: np.ndarray, SLM: np.ndarray, delta3x: float,
                delta3y: float, delta4x: float, delta4y: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase
        retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3x: Apparent sampling size of the SLM as seen from the
        sensor plane along x
        :param delta3y: Apparent sampling size of the SLM as seen from the
        sensor plane along y
        :param delta4x: Sampling size of the sensor plane along x
        :param delta4y: Sampling size of the sensor plane along y
        :return u3_est, u4_est, idx_converge: Estimated fields of size (Nx,Ny)
        and the convergence indices to check
        convergence speed
        """
        wvl = self.wavelength
        z3 = self.z
        # parameters
        Nx = y0.shape[1]
        Ny = y0.shape[0]
        N_batch = self.N_mod
        N_os = self.N_os
        N_iter = self.N_gs
        u3_batch = cp.zeros((Ny, Nx, N_os), dtype=cp.complex64)
        u4 = cp.zeros((Ny, Nx, N_os), dtype=cp.complex64)
        y = cp.zeros((Ny, Nx, N_os), dtype=cp.complex64)
        # initilize a3
        k = self.n * 2 * np.pi / wvl
        xx = cp.linspace(0, Nx - 1, Nx, dtype=cp.float32) - (Nx / 2) *\
            cp.ones(Nx, dtype=cp.float32)
        yy = cp.linspace(0, Ny - 1, Ny, dtype=cp.float32) - (Ny / 2) *\
            cp.ones(Ny, dtype=cp.float32)
        X, Y = float(delta4x) * cp.meshgrid(xx, yy)[0], float(delta4y) *\
            cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j*(k/(2*z3))*R**2)
        del xx, yy, X, Y, R
        SLM_batch = cp.asarray(SLM[:, :, 0])
        for ii in range(N_os):
            y0_batch = cp.asarray(y0[:, :, ii])
            u3_batch[:, :, ii] = self.frt_gpu_s(y0_batch/Q, delta4x, delta4y,
                                                self.wavelength, -z3)
            u3_batch[:, :, ii] *= cp.conj(SLM_batch)
        u3 = cp.mean(u3_batch, 2)
        # i_mask, j_mask = self.define_mask(np.abs(y0[:, :, 0]) ** 2,
        # plot=True)[1:3]
        # Recon run : GS loop
        idx_converge = np.empty(N_iter)
        for jj in range(N_iter):
            sys.stdout.flush()
            u3_collect = cp.zeros(u3.shape, dtype=cp.complex64)
            idx_converge0 = np.empty(N_batch)
            for idx_batch in range(N_batch):
                # put the correct batch into the GPU
                SLM_batch = SLM[:, :, idx_batch]
                y0_batch = y0[:, :,
                              int(N_os * idx_batch):int(N_os*(idx_batch+1))]
                y0_batch = cp.asarray(y0_batch)
                SLM_batch = cp.asarray(SLM_batch)
                for _ in range(N_os):
                    u4[:, :, _] = self.frt_gpu_s(u3 * SLM_batch, delta3x,
                                                 delta3y, self.wavelength, z3)
                    # impose the amplitude
                    y[:, :, _] = y0_batch[:, :, _] *\
                        cp.exp(1j * cp.angle(u4[:, :, _]))
                    # [:,:,_] = u4[:,:,_]
                    # y[i_mask:j_mask,i_mask:j_mask,_] = y0_batch[
                    # i_mask:j_mask,i_mask:j_mask,_] \
                    # *cp.exp(1j * cp.angle(u4[i_mask:j_mask,i_mask:j_mask,_]))
                    u3_batch[:, :, _] = self.frt_gpu_s(
                        y[:, :, _], delta4x, delta4y, self.wavelength, -z3) *\
                        cp.conj(SLM_batch)
                # add U3 from each batch
                u3_collect = u3_collect + cp.mean(u3_batch, 2)
                # convergence index matrix for each batch
                idx_converge0[idx_batch] = (1/np.sqrt(Nx*Ny)) * \
                    cp.linalg.norm((cp.abs(u4)-(1/(Nx*Ny)) *
                                    cp.sum(cp.abs(SLM_batch)) *
                                    y0_batch)*(y0_batch > 0))
                # eventual mask absorption
            u3 = (u3_collect / N_batch)  # average over batches
            idx_converge[jj] = np.mean(idx_converge0)  # sum over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")

            # exit if the matrix doesn 't change much
            if jj > 1:
                eps = cp.abs(idx_converge[jj]-idx_converge[jj-1]) / \
                    idx_converge[jj]
                if eps < 1e-4:
                    # if cp.abs(idx_converge[jj]) < 5e-3:
                    # if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    # idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj])
                    break
        # propagate solution to sensor plane
        u4_est = self.frt_gpu_s(u3, delta3x, delta3y, self.wavelength, z3) * Q
        return u3, u4_est, idx_converge

    def do_GS_step(self, y0: np.ndarray, SLM: np.ndarray, U3: np.ndarray,
                    u3: np.ndarray, delta3x: float, delta3y: float, 
                    delta4x: float, delta4y: float, plan_fft=None):
        """Does one step of the GS loop in place

        Args:
            y0 (np.ndarray): Target intensity vector
            SLM (np.ndarray): SLM patterns
            U3 (np.ndarray): Circulating fields
            u3 (np.ndarray): Current estimation
            delta3x (float): SLM plane pitch along x
            delta3y (float): SLM plane pitch along y
            delta4x (float): Camera plane pitch along x
            delta4y (float): Camera plane pitch along y
            plan_fft (cufft_fft_plan, optional): FFT Plan. Defaults to None.
        """
        tpb = (1, 16, 16)
        bpg = (1, SLM.shape[0] // tpb[0], SLM.shape[1] // tpb[1])
        U3 = SLM * u3
        self.frt_gpu_vec_s(U3, delta3x, delta3y,
                                self.wavelength, self.z, plan=plan_fft)
        # U3 = y0 * cp.exp(1j*cp.angle(U3))
        ker_impose_amp(bpg, tpb, (y0, U3, U3, U3.shape[0], U3.shape[1], U3.shape[2]))
        # ker_impose_amp(y0, U3)
        self.frt_gpu_vec_s(U3, delta4x, delta4y, self.wavelength,
                                -self.z, plan=plan_fft)
        U3 *= cp.conj(SLM)
        # ker_multiply_conjugate(SLM, U3)
        # ker_multiply_conjugate(bpg, tpb, (SLM, U3, U3.shape[0], U3.shape[1], U3.shape[2]))
        u3[:] = cp.mean(U3, 0)  # average over batches

    def do_CG_step(self, jj: int, u3_new: np.ndarray, y0: np.ndarray, SLM: np.ndarray, U3: np.ndarray,
                    u3: np.ndarray, delta3x: float, delta3y: float, 
                    delta4x: float, delta4y: float, plan_fft=None):
        """Does one step of the GS loop in place

        Args:
            jj (int) : iteration number
            u3_new = cp.empty_like(u3)
            y0 (np.ndarray): Target intensity vector
            SLM (np.ndarray): SLM patterns
            U3 (np.ndarray): Circulating fields
            u3 (np.ndarray): Current estimation
            delta3x (float): SLM plane pitch along x
            delta3y (float): SLM plane pitch along y
            delta4x (float): Camera plane pitch along x
            delta4y (float): Camera plane pitch along y
            plan_fft (cufft_fft_plan, optional): FFT Plan. Defaults to None.
        """
        # apply modulation
        U3 = SLM * u3
        # propagate to image field
        self.frt_gpu_vec_s(U3, delta3x, delta3y,
                                self.wavelength, self.z, plan=plan_fft)
        # compute error
        err = cp.mean(cp.linalg.norm(cp.abs(U3)-cp.abs(y0), axis=(0, 1))**2)*1/(U3.shape[0]*U3.shape[1])
        # impose amplitude constraint
        U3 = y0 * cp.exp(1j*cp.angle(U3))
        # back propagate
        self.frt_gpu_vec_s(U3, delta4x, delta4y, self.wavelength,
                                -self.z, plan=plan_fft)
        # remove modulation
        U3 *= cp.conj(SLM)
        # reduction 
        u3_new[:] = cp.mean(U3, 0) 
        if jj == 0:
            D = u3_new - u3
        else:
            D = u3_new - u3 + (err/self.err_old)*self.D_old
        # update guess while adding gradient
        # u3[:] = u3_new + self.hk*(u3_new-self.u3_old)
        u3[:] = u3_new + self.hk*D
        self.u3_old[:] = u3_new
        self.D_old[:] = D
        self.err_old = err
        return err

    def do_CG_step_fast(self, jj: int, u3_new: np.ndarray, y0: np.ndarray, SLM: np.ndarray, U3: np.ndarray,
                    u3: np.ndarray, delta3x: float, delta3y: float, 
                    delta4x: float, delta4y: float, plan=None):
        """Does one step of the GS loop in place

        Args:
            jj (int) : iteration number
            u3_new = cp.empty_like(u3)
            y0 (np.ndarray): Target intensity vector
            SLM (np.ndarray): SLM patterns
            U3 (np.ndarray): Circulating fields
            u3 (np.ndarray): Current estimation
            delta3x (float): SLM plane pitch along x
            delta3y (float): SLM plane pitch along y
            delta4x (float): Camera plane pitch along x
            delta4y (float): Camera plane pitch along y
            plan_fft (cufft_fft_plan, optional): FFT Plan. Defaults to None.
        """
        norm_f = (delta3x*delta3y)/(1j * self.wavelength * self.z)
        norm_i = (delta4x*delta4y)/(-1j * self.wavelength * self.z)
        # apply modulation
        U3 = SLM * u3
        # propagate to image field
        plan.fft(U3, U3, cp.cuda.cufft.CUFFT_FORWARD) 
        # compute error
        err = cp.mean(cp.linalg.norm(cp.abs(U3*norm_f)-cp.abs(y0), axis=(0, 1))**2)*1/(U3.shape[0]*U3.shape[1])
        # impose amplitude constraint
        U3[:] = ker_impose_amp_norm(y0, U3, norm_f)
        # back propagate
        plan.fft(U3, U3, cp.cuda.cufft.CUFFT_INVERSE) 
        # reduction 
        u3_new[:] = ker_multiply_conjugate_sum_norm(SLM, U3, (norm_i/U3.shape[0]))
        # if jj == 0:
        #     D = u3_new - u3
        # else:
        #     D = u3_new - u3 + (err/self.err_old)*self.D_old
        # update guess while adding gradient
        u3[:] = u3_new + self.hk*(u3_new-self.u3_old)
        # u3[:] = u3_new + self.hk*D
        self.u3_old[:] = u3_new
        # self.D_old[:] = D
        # self.err_old = err
        return 1.0

#TODO(Tangui) Rewrite the GS_step in CUDA C++ :'( :'( :'( 
    def do_GS_step_fast(self, y0: cp.ndarray, SLM: cp.ndarray, U3: np.ndarray,
                    u3: cp.ndarray, delta3x: cp.float32, delta3y: cp.float32, 
                    delta4x: cp.float32, delta4y: cp.float32, plan):  
        """Does one step of the GS loop in place, faster provided you have a plan ...

        Args:
            y0 (np.ndarray): Target intensity vector
            SLM (np.ndarray): SLM patterns
            U3 (np.ndarray): Circulating fields
            u3 (np.ndarray): Current estimation
            delta3x (float): SLM plane pitch along x
            delta3y (float): SLM plane pitch along y
            delta4x (float): Camera plane pitch along x
            delta4y (float): Camera plane pitch along y
            plan_fft (cufft_fft_plan): FFT Plan. 
        """
        norm_f = (delta3x*delta3y)/(1j * self.wavelength * self.z)
        norm_i = (delta4x*delta4y)/(-1j * self.wavelength * self.z)
        U3[:] = SLM * u3
        plan.fft(U3, U3, cp.cuda.cufft.CUFFT_FORWARD) 
        U3[:] = ker_impose_amp_norm(y0, U3, norm_f)
        plan.fft(U3, U3, cp.cuda.cufft.CUFFT_INVERSE) 
        u3[:] = ker_multiply_conjugate_sum_norm(SLM, U3, (norm_i/U3.shape[0]))


    def WISHrun_vec(self, y0: np.ndarray, SLM: np.ndarray, delta3x: float,
                    delta3y: float, delta4x: float, delta4y: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase
        retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3: Apparent sampling size of the SLM as seen from the
        sensor plane
        :param delta4: Sampling size of the sensor plane
        :return u3_est, u4_est, idx_converge: Estimated fields of size (Ny,Nx)
        and the convergence indices to check convergence speed
        """
        
        wvl = self.wavelength
        # parameters
        Nx = y0.shape[1]
        Ny = y0.shape[0]
        Nim = self.Nim
        N_os = self.N_os
        N_iter = self.N_gs
        k = 2 * np.pi / wvl
        xx = cp.linspace(0, Nx - 1, Nx, dtype=cp.float32) - \
            (Nx / 2) * cp.ones(Nx, dtype=cp.float32)
        yy = cp.linspace(0, Ny - 1, Ny, dtype=cp.float32) - \
            (Ny / 2) * cp.ones(Ny, dtype=cp.float32)
        X, Y = float(delta4x) * cp.meshgrid(xx, yy)[0], \
            float(delta4y) * cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j * (k / (2 * self.z)) * R ** 2)
        del xx, yy, X, Y, R
        U3 = cp.empty((Nim, Ny, Nx), dtype=cp.complex64)
        SLM = cp.asarray(SLM.repeat(N_os, axis=2), dtype=cp.complex64)
        y0 = cp.asarray(y0, dtype=cp.complex64)
        SLM = SLM.transpose(2, 0, 1)
        y0 = y0.transpose(2, 0, 1)
        for ii in range(self.Nim):
            y0_batch = y0[ii, :, :]
            SLM_batch = SLM[ii, :, :]
            U3[ii, :, :] = self.frt_gpu_s(y0_batch / Q, delta4x, delta4y,
                                          self.wavelength, -self.z) *\
                cp.conj(SLM_batch)  # y0_batch gpu
        u3 = cp.mean(U3, 0)
        u30 = cp.copy(u3)
        del SLM_batch, y0_batch
        # GS loop
        idx_converge = np.empty(N_iter//5)
        idx_converge_gc = np.empty(N_iter, dtype=cp.float32)
        plan_fft = get_fft_plan(U3, axes=(1, 2), value_type='C2C')
        # gpu_frt(cp.empty((3,3,3), dtype=cp.complex64), 1.0, 1.0, 1.0, 1.0, plan_fft)
        T_run_0 = time.time()
        # fftshift first 
        SLM = cp.fft.ifftshift(SLM, axes=(1, 2))
        u3 = cp.fft.ifftshift(u3)
        y0 = cp.fft.fftshift(y0, axes=(1, 2))
        # t_exec_gpu = cp.empty(N_iter, dtype=cp.float32)
        # start_gpu0 = cp.cuda.Event()
        # end_gpu0 = cp.cuda.Event()
        # start_gpu0.record()
        u3_new = cp.empty_like(u3) # for CG
        self.u3_old = u3.copy() # for CG
        self.hk = 1.0 # for CG
        self.err_old = 1.0
        self.D_old = np.empty_like(u3)
        with cp.cuda.profile():
            for jj in range(N_iter):
                # sys.stdout.flush()
                # start_gpu = cp.cuda.Event()
                # end_gpu = cp.cuda.Event()
                # start_gpu.record()
                # self.do_GS_step(y0, SLM, u3, delta3x, delta3y, delta4x, delta4y,
                #                 plan_fft=plan_fft)
                # self.do_GS_step_fast(y0, SLM, U3, u3, delta3x, delta3y, delta4x, delta4y,
                #                      plan_fft)
                # sys.stdout.write(f"\rGS iteration {jj+1}/{N_iter}")
                err = self.do_CG_step(jj, u3_new, y0, SLM, U3, u3, delta3x, delta3y, delta4x, delta4y,
                        plan_fft)
                idx_converge_gc[jj] = err
                sys.stdout.write(f"\rGS iteration {jj+1}/{N_iter}  err = {err}")
                if jj > 0 and jj%100==0:
                    self.hk *= 0.75
                eps = cp.abs(idx_converge_gc[jj] - idx_converge_gc[jj-1])/idx_converge_gc[jj]
                if eps < 1e-5:
                    idx_converge_gc = idx_converge_gc[0:jj]
                    break
                # if idx_converge_gc[jj] >= idx_converge_gc[jj-10]:
                #     self.hk *= 0.9
                # end_gpu.record()
                # end_gpu.synchronize()
                # t_exec_gpu[jj] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)*1e-3
                # norm_f = (delta3x*delta3y)/(1j * self.wavelength * self.z)
                # norm_i = (delta4x*delta4y)/(-1j * self.wavelength * self.z)
                # U3[:] = SLM * u3
                # plan_fft.fft(U3, U3, cp.cuda.cufft.CUFFT_FORWARD)
                # U3 *= norm_f
                # # convergence index matrix for every 5 iterations
                # if jj % 5 == 0:
                #     # idx_converge0 = (1 / np.sqrt(Nx*Ny)) * \
                #     #     cp.linalg.norm((cp.abs(U3)-y0) *
                #     #                    (y0 > 0), axis=(0, 1))
                #     idx_converge[jj//5] = cp.mean(cp.linalg.norm(cp.abs(U3)-cp.abs(y0), axis=(0, 1))**2)*1/(U3.shape[0]*U3.shape[1])
                #     prt = f"\rGS iteration {jj + 1}  (convergence index : {idx_converge[jj//5]})"
                #     sys.stdout.write(prt)
                # sys.stdout.write(f"\rGS iteration {jj + 1}")
                # # exit if the matrix doesn't change much
                # if (jj > 1) & (jj % 5 == 0):
                #     eps = cp.abs(idx_converge[jj//5] - idx_converge[jj//5 - 1]) / \
                #         idx_converge[jj//5]
                #     if eps < 1e-5:
                #         print('\nConverged. Exit the GS loop ...')
                #         # idx_converge = idx_converge[0:jj]
                #         idx_converge = cp.asnumpy(idx_converge[0:jj//5])
                #         break
                # U3[:] = ker_impose_amp_norm(y0, U3, 1.0)
                # plan_fft.fft(U3, U3, cp.cuda.cufft.CUFFT_INVERSE) 
                # u3[:] = ker_multiply_conjugate_sum_norm(SLM, U3, (norm_i/U3.shape[0]))
                # if jj == N_iter-1:
                #     print('\nMax iteration number reached. Exit ...')
            # optional profiling
            # t_exec = repeat_c(self.do_GS_step, (y0, SLM, u3, delta3x, delta3y, delta4x, delta4y,
            #                     plan_fft), n_repeat=N_iter)
            # t_exec_cpu = t_exec.cpu_times
            # t_exec_gpu = t_exec.gpu_times[0, :]
            # fftshift
            u3 = cp.fft.fftshift(u3)
        # propagate solution to sensor plane
        T_run = time.time()-T_run_0
        # end_gpu0.record()
        # end_gpu0.synchronize()
        # T_run_gpu = cp.cuda.get_elapsed_time(start_gpu0, end_gpu0)*1e-3
        u4_est = self.frt_gpu_s(u3, delta3x, delta3y, self.wavelength, self.z) * Q
        print(f"\nTime spent in the GS loop : {T_run} s")
        # print(f"GPU time spent in the GS loop : {T_run_gpu} s")
        # print(f"GPU per iteration time {cp.mean(t_exec_gpu)} +/- {cp.std(t_exec_gpu)} s")
        # plt.plot(t_exec_cpu)
        # plt.plot(t_exec_gpu)
        # plt.plot(np.cumsum(t_exec_cpu))
        # plt.plot(np.cumsum(t_exec_gpu))
        # plt.plot(np.cumsum(t_exec_cpu+t_exec_gpu))
        # plt.ylabel("Time in s")
        # plt.xlabel("Iteration")
        # plt.title("Run time")
        # plt.yscale("log")
        # plt.legend(["CPU", "GPU", "Cumulative CPU", "Cumulative GPU", "Cumulative tot"])
        return u3, u4_est, idx_converge_gc

#TODO(Tangui) Wrap WISH_measurement into a proper camera object
class WISH_Camera_gpu(WISH_Sensor):
    def __init__(self, cam, slm):
        # self.frame_buffer = ... (efficient way to store / reuse a frame buffer)
        return
    
    def alignment(self):
        # something smart to auto align with an alignment target pattern ?
        # extract affine transform from target pattern deformation as explained in OpenCV doc
        # https://learnopencv.com/camera-calibration-using-opencv/
        return
    
    def capture_ims(self):
        # should check DMD or SLM model, Camera model to reuse one or the other function
        # capture_ims_flir or capture_ims_XXX
        return

    def run(self):
        # runs in an infinite loop to update the frame_buffer
        # what callbacks should I need ?
        return
    

class WISH_Sensor_cpu:
    def __init__(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path)
        self.d_SLM = float(conf["params"]["d_SLM"])
        self.d_CAM = float(conf["params"]["d_CAM"])
        self.wavelength = float(conf["params"]["wavelength"])
        # propagation distance
        self.z = float(conf["params"]["z"])
        # number of GS iterations
        self.N_gs = int(conf["params"]["N_gs"])
        # number of modulation steps
        self.N_mod = int(conf["params"]["N_mod"])
        # number of observations per image (to avg noise)
        self.N_os = int(conf["params"]["N_os"])
        self.Nim = self.N_mod * self.N_os
        # intensity threshold for the signal region
        self.threshold = float(conf['params']['mask_threshold'])
        self.noise = float(conf['params']['noise'])

    def define_mask(self, I: np.ndarray, plot: bool = False):
        """
        A function to define the signal region automatically from the provided
        intensity and threshold
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
            k, m = i_min, i_max
        else:
            mask_sr[j_min:j_max, j_min:j_max] = 1
            k, m = j_min, j_max
        if plot:
            fig = plt.figure(0)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            im1 = ax1.imshow(I, cmap="viridis")
            ax1.set_title("Intensity")
            im2 = ax2.imshow(mask_sr, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max],
                               color='r')
            scat.set_label('Threshold point')
            ax2.legend()
            fig.colorbar(im1, cax=cax1)
            fig.colorbar(im2, cax=cax2)
            plt.show()
        return mask_sr, k, m

    def crop_center(self, img: np.ndarray, cropx: int, cropy: int):
        """
        A function to crop around the center of an array
        :param img: Array to crop
        :param cropx: Size along x direction of cropped array
        :param cropy: Size along y direction of cropped array
        :return: cropped array
        """
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    @staticmethod
    def modulate(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too
        much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        # random matrix between [0 , 1]
        M = (1/256)*np.random.random_integers(0, 255, (h, w))
        M = M.astype(np.float32)
        phi_m = zoom(M, (shape[0]/M.shape[0], shape[1]/M.shape[1]))
        return phi_m

    @staticmethod
    def modulate_binary(shape: tuple, pxsize: int = 10):
        """
        A function to randomly modulating a phase map without introducing too
        much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """

        h, w = int(shape[0] / pxsize), int(shape[1] / pxsize)
        # random intensity mask
        M = np.random.choice(np.asarray([0, 0.5]), (h, w))
        phi_m = zoom(M, (shape[0] / M.shape[0], shape[1] / M.shape[1]),
                     order=0)
        return phi_m

    def gaussian_profile(self, I: np.ndarray, sigma: float):
        """
        Applies a gaussian profile to the intensity provided
        :param I: Intensity to which a gaussian profile is going to be applied
        :param sigma: Standard deviation of the gaussian profile, in fraction
        of the provided intensity size
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
    def frt(A0: np.ndarray, d1x: float, d1y: float, wv: float,
            z: float):
        """
        Implements propagation using Fresnel diffraction.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: index of refraction
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        k = 2 * np.pi / wv
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        x = np.linspace(0, Nx - 1, Nx, dtype=np.float32) -\
            (Nx / 2) * np.ones(Nx, dtype=np.float32)
        y = np.linspace(0, Ny - 1, Ny, dtype=np.float32) -\
            (Ny / 2) * np.ones(Ny, dtype=np.float32)
        d2x = wv * abs(z) / (Nx * d1x)
        d2y = wv * abs(z) / (Ny * d1y)
        X1, Y1 = d1x * np.meshgrid(x, y)[0], d1y * np.meshgrid(x, y)[1]
        X2, Y2 = d2x * np.meshgrid(x, y)[0], d2y * np.meshgrid(x, y)[1]
        R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        A0 = A0 * np.exp(1j * (k / (2 * z)) * R1 ** 2)
        if z > 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fft2(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(0, 1))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.ifft2(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(0, 1))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * np.exp(1j * (k / (2 * z)) * R2 ** 2)
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_vec(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                z: float):
        """
        Implements propagation using Fresnel diffraction. Vectorized.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: index of refraction
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        k = 2 * np.pi / wv
        Nx = A0.shape[2]
        Ny = A0.shape[1]
        x = np.linspace(0, Nx - 1, Nx, dtype=np.float32) -\
            (Nx / 2) * np.ones(Nx, dtype=np.float32)
        y = np.linspace(0, Ny - 1, Ny, dtype=np.float32) -\
            (Ny / 2) * np.ones(Ny, dtype=np.float32)
        d2x = wv * abs(z) / (Nx * d1x)
        d2y = wv * abs(z) / (Ny * d1y)
        X1, Y1 = d1x * np.meshgrid(x, y)[0], d1y * np.meshgrid(x, y)[1]
        X2, Y2 = d2x * np.meshgrid(x, y)[0], d2y * np.meshgrid(x, y)[1]
        R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        A0 = A0 * np.exp(1j * (k / (2 * z)) * R1 ** 2)
        if z > 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(1, 2))
            A0 = pyfftw.interfaces.numpy_fft.fft2(A0, axes=(1, 2))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(1, 2))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(1, 2))
            A0 = pyfftw.interfaces.numpy_fft.ifft2(A0, axes=(1, 2))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(1, 2))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * np.exp(1j * (k / (2 * z)) * R2 ** 2)
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_s(A0: np.ndarray, d1x: float, d1y: float, wv: float, z: float):
        """
        Simplified Fresnel propagation optimized for fast CPU computing.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        if z > 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fft2(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(0, 1))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = pyfftw.interfaces.numpy_fft.ifftshift(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.ifft2(A0, axes=(0, 1))
            A0 = pyfftw.interfaces.numpy_fft.fftshift(A0, axes=(0, 1))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    @staticmethod
    def frt_vec_s(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                  z: float, fft: pyfftw.FFTW = None):
        """
        Simplified Fresnel propagation optimized for fast CPU computing.
        Vectorized
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        Nx = A0.shape[2]
        Ny = A0.shape[1]
        if z > 0:
            if fft is None:
                A0 = mkl_fft.fft2(A0, axes=(1, 2), overwrite_x=True)
            else:
                A0 = fft(A0)
            A0 = d1x*d1y * A0

        elif z <= 0:
            if fft is None:
                A0 = mkl_fft.ifft2(A0, axes=(1, 2), overwrite_x=True)
            else:
                A0 = fft(A0)
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * 1 / (1j * wv * z)
        return A0

    def u4Tou3(self, u4: np.ndarray, delta4x: float, delta4y: float,
               z3: float):
        """
        Propagates back a field from the sensor plane to the SLM plane
        :param u4: Field to propagate back
        :param delta4x: Sampling size of the field u4 in the x direction
        :param delta4y: Sampling size of the field u4 in the y direction
        :param z3: Propagation distance in metres
        :return: u3 the back propagated field
        """
        u3 = self.frt(u4, delta4x, delta4y, self.wavelength, -z3)
        return u3

    def process_SLM(self, slm: np.ndarray, Nx: int, Ny: int, delta4x: float,
                    delta4y: float, type: str):
        """
        Scales the pre submitted SLM plane field (either amplitude of phase) to
        the right size taking into account the apparent size of the SLM in the
        sensor field of view.
        :param slm: Input SLM patterns
        :param Nx: Size of the calculation in x
        :param Ny: Size of the calculation in y
        :param delta3x: Sampling size of the SLM plane (typically the
        "apparent" sampling size wvl*z/Nx*d_Sensorx )
        :param delta3y: Sampling size of the SLM plane (typically the
        "apparent" sampling size wvl*z/Ny*d_Sensory )
        :param type : "amp" / "phi" amplitude or phase pattern.
        :return SLM: Rescaled and properly shaped SLM patterns of size
        (Ny,Nx,N_batch)
        """
        delta_SLM = self.d_SLM
        N_batch = self.N_mod
        delta3x = self.wavelength * self.z / (Nx * delta4x)
        delta3y = self.wavelength * self.z / (Ny * delta4y)
        if slm.dtype == 'uint8':
            slm = slm.astype(np.float32)/256.
        # check if SLM can be centered in the computational window
        Nxslm = slm.shape[1]
        Nyslm = slm.shape[0]
        cdx = (np.round(Nxslm*delta_SLM/delta3x) % 2) != 0
        cdy = (np.round(Nyslm*delta_SLM/delta3y) % 2) != 0
        if cdx or cdy:
            Z = np.linspace(self.z-2e-5, self.z+2e-5, int(1e2),
                            dtype=np.float32)
            D3x = self.wavelength * Z / (Nx * self.d_CAM)
            D3y = self.wavelength * Z / (Ny * self.d_CAM)
            X = np.round(Nxslm*self.d_SLM/D3x)
            Y = np.round(Nyslm*self.d_SLM/D3y)
            X = X % 2
            Y = Y % 2
            diff = X+Y
            z_corr = Z[diff == np.min(diff)][0]
            print(
                "WARNING : Propagation is such that the SLM cannot be" +
                " centered in the computational window. Distance will be set" +
                " to closest matching distance" +
                f" z = {np.round(z_corr*1e3, decimals=2)} mm.")
            print("\nPlease adjust propagation distance or continue.")
            cont = None
            while cont is None:
                cont = input("\nContinue ? [y/n]")
                if cont == 'y':
                    self.z = z_corr
                elif cont == 'n':
                    exit()
                else:
                    cont = None
        delta3x = self.wavelength * self.z / (Nx * delta4x)
        delta3y = self.wavelength * self.z / (Ny * delta4y)
        if slm.ndim == 3:
            slm3 = np.empty((Ny, Nx, N_batch), dtype=np.float32)
            # scale SLM slices to the right size
            for i in range(N_batch):
                slm1 = zoom(np.asarray(slm[:, :, i]),
                            [delta_SLM/delta3y, delta_SLM/delta3x], order=0)
                if slm1.shape[0] > Ny and slm1.shape[1] <= Nx:
                    slm1 = self.crop_center(slm1, slm1.shape[1], Ny)
                elif slm1.shape[0] <= Ny and slm1.shape[1] > Nx:
                    slm1 = self.crop_center(slm1, Nx, slm1.shape[0])
                elif slm1.shape[0] > Ny and slm1.shape[1] > Nx:
                    slm1 = self.crop_center(slm1, Nx, Ny)
                if slm1.shape[0] < Ny or slm1.shape[1] < Nx:
                    pady = int(np.ceil((Ny - slm1.shape[0]) / 2))
                    padx = int(np.ceil((Nx - slm1.shape[1]) / 2))
                    slm1 = np.pad(slm1, ((pady, pady), (padx, padx)))
                slm3[:, :, i] = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3.astype(np.complex64)
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        elif slm.ndim == 2:
            slm2 = slm
            slm3 = np.empty((Ny, Nx), dtype=np.float32)
            # scale SLM slices to the right size
            slm1 = zoom(slm2, [delta_SLM / delta3y, delta_SLM / delta3x])
            slm1 = np.pad(slm1, (int(np.ceil((Ny - slm1.shape[0]) / 2)),
                                 int(np.ceil((Nx - slm1.shape[1]) / 2))))
            slm3 = slm1
            if type == "phi":
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
            elif type == "amp":
                SLM = slm3.astype(np.complex64)
            else:
                print("Wrong type specified : type can be 'amp' or 'phi' ! ")
                raise
        return SLM

    def gen_ims(self, u3: np.ndarray, slm: np.ndarray, z3: float,
                delta3x: float, delta3y: float, noise: float):
        """
        Generates dummy signal in the sensor plane from the pre generated SLM
        patterns
        :param u3: Initial field in the SLM plane
        :param phi0 : Initial phase typically the calibration of the SLM
        :param slm : Pre generated slm patterns
        :param z3: Propagation distance in metres
        :param delta3x: "apparent" sampling size of the SLM plane (as seen by
        the image plane from z3 m away) in the x direction
        :param delta3y: "apparent" sampling size of the SLM plane (as seen by
        the image plane from z3 m away) in the x direction
        :param noise: Intensity of the gaussian noise added to the images
        :return ims: Generated signal in the sensor plane of size (N,N,Nim)
        """
        Nx = u3.shape[1]
        Ny = u3.shape[0]
        Nim = self.Nim
        N_os = self.N_os
        delta_SLM = self.d_SLM
        Lx_SLM = delta_SLM * slm.shape[1]
        Ly_SLM = delta_SLM * slm.shape[0]
        x = np.linspace(0, Nx - 1, Nx, dtype=np.float32) -\
            (Nx / 2) * np.ones(Nx, dtype=np.float32)
        y = np.linspace(0, Ny - 1, Ny, dtype=np.float32) -\
            (Ny / 2) * np.ones(Ny, dtype=np.float32)
        XX, YY = np.meshgrid(x, y)
        A_SLM = (np.abs(XX) * delta3x < Lx_SLM / 2) * \
                (np.abs(YY) * delta3y < Ly_SLM / 2)

        if slm.dtype == 'uint8':
            slm = slm.astype(np.float32)/256
        ims = np.zeros((Ny, Nx, Nim), dtype=np.float32)
        for i in range(Nim):
            sys.stdout.write(f"\rGenerating image {i+1} out of {Nim} ...")
            sys.stdout.flush()
            a31 = u3 * A_SLM * slm[:, :, i//N_os]
            a4 = self.frt(a31, delta3x, delta3y, self.wavelength, z3)
            w = noise * np.random.standard_normal((Ny, Nx,))
            ya = np.abs(a4)**2 + w
            ya[ya < 0] = 0
            # ya = shift(ya, (1*np.random.standard_normal(1),
            #           1*np.random.standard_normal(1)))
            ims[:, :, i] = ya
            del a31, a4, ya
        return ims

    def process_ims(self, ims: np.ndarray, Nx: int, Ny: int):
        """
        Converts images to amplitudes and eventually resizes them.
        :param ims: images to convert
        :param Nx: Pixel nbr of the sensor along x
        :param Ny: Pixel nbr of the sensor along y
        :return y0 : Processed field of size (Nx,Ny, Nim)
        """
        if ims.dtype == 'uint8':
            ims = (ims/256).astype(np.float32)
        y0 = np.real(np.sqrt(ims))  # change from intensity to magnitude
        return y0.astype(np.float32)

    def WISHrun(self, y0: np.ndarray, SLM: np.ndarray, delta3x: float,
                delta3y: float, delta4x: float, delta4y: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase
        retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3x: Apparent sampling size of the SLM as seen from the
        sensor plane along x
        :param delta3y: Apparent sampling size of the SLM as seen from the
        sensor plane along y
        :param delta4x: Sampling size of the sensor plane along x
        :param delta4y: Sampling size of the sensor plane along y
        :return u3_est, u4_est, idx_converge: Estimated fields of size (Nx,Ny)
        and the convergence indices to check
        convergence speed
        """
        wvl = self.wavelength
        z3 = self.z
        # parameters
        Nx = y0.shape[1]
        Ny = y0.shape[0]
        N_batch = self.N_mod
        N_os = self.N_os
        N_iter = self.N_gs
        u3_batch = np.zeros((Ny, Nx, N_os), dtype=np.complex64)
        u4 = np.zeros((Ny, Nx, N_os), dtype=np.complex64)
        y = np.zeros((Ny, Nx, N_os), dtype=np.complex64)
        # initilize a3
        k = self.n * 2 * np.pi / wvl
        xx = np.linspace(0, Nx - 1, Nx, dtype=np.float32) - (Nx / 2) *\
            np.ones(Nx, dtype=np.float)
        yy = np.linspace(0, Ny - 1, Ny, dtype=np.float32) - (Ny / 2) *\
            np.ones(Ny, dtype=np.float32)
        X, Y = float(delta4x) * np.meshgrid(xx, yy)[0], float(delta4y) *\
            np.meshgrid(xx, yy)[1]
        R = np.sqrt(X ** 2 + Y ** 2)
        Q = np.exp(1j*(k/(2*z3))*R**2)
        del xx, yy, X, Y, R
        SLM_batch = SLM[:, :, 0]
        for ii in range(N_os):
            y0_batch = np.asarray(y0[:, :, ii])
            u3_batch[:, :, ii] = self.frt_s(y0_batch/Q, delta4x, delta4y,
                                            self.wavelength, -z3)
            u3_batch[:, :, ii] *= np.conj(SLM_batch)
        u3 = np.mean(u3_batch, 2)
        # i_mask, j_mask = self.define_mask(np.abs(y0[:, :, 0]) ** 2,
        # plot=True)[1:3]
        # GS loop
        idx_converge = np.empty(N_iter)
        for jj in range(N_iter):
            sys.stdout.flush()
            u3_collect = np.zeros(u3.shape, dtype=np.complex64)
            idx_converge0 = np.empty(N_batch)
            for idx_batch in range(N_batch):
                SLM_batch = SLM[:, :, idx_batch]
                y0_batch = y0[:, :,
                              int(N_os * idx_batch):int(N_os*(idx_batch+1))]
                y0_batch = np.asarray(y0_batch)
                SLM_batch = np.asarray(SLM_batch)
                for _ in range(N_os):
                    u4[:, :, _] = self.frt_s(u3 * SLM_batch, delta3x,
                                             delta3y, self.wavelength, z3)
                    # impose the amplitude
                    y[:, :, _] = y0_batch[:, :, _] *\
                        np.exp(1j * np.angle(u4[:, :, _]))
                    # [:,:,_] = u4[:,:,_]
                    # y[i_mask:j_mask,i_mask:j_mask,_] = y0_batch[
                    # i_mask:j_mask,i_mask:j_mask,_] \
                    # *np.exp(1j * np.angle(u4[i_mask:j_mask,i_mask:j_mask,_]))
                    u3_batch[:, :, _] = self.frt_s(
                        y[:, :, _], delta4x, delta4y, self.wavelength, -z3) *\
                        np.conj(SLM_batch)
                # add U3 from each batch
                u3_collect = u3_collect + np.mean(u3_batch, 2)
                # convergence index matrix for each batch
                idx_converge0[idx_batch] = (1/np.sqrt(Nx*Ny)) * \
                    np.linalg.norm((np.abs(u4)-(1/(Nx*Ny)) *
                                    np.sum(np.abs(SLM_batch)) *
                                    y0_batch)*(y0_batch > 0))
                # eventual mask absorption
            u3 = (u3_collect / N_batch)  # average over batches
            idx_converge[jj] = np.mean(idx_converge0)  # sum over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")

            # exit if the matrix doesn 't change much
            if jj > 1:
                eps = np.abs(idx_converge[jj]-idx_converge[jj-1]) / \
                    idx_converge[jj]
                if eps < 1e-4:
                    # if np.abs(idx_converge[jj]) < 5e-3:
                    # if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    # idx_converge = idx_converge[0:jj]
                    idx_converge = idx_converge[0:jj]
                    break
        # propagate solution to sensor plane
        u4_est = self.frt_s(u3, delta3x, delta3y, self.wavelength, z3) * Q
        return u3, u4_est, idx_converge

    def WISHrun_vec(self, y0: np.ndarray, SLM: np.ndarray, delta3x: float,
                    delta3y: float, delta4x: float, delta4y: float):
        """
        Runs the WISH algorithm using a Gerchberg Saxton loop for phase
        retrieval.
        :param y0: Target modulated amplitudes in the sensor plane
        :param SLM: SLM modulation patterns
        :param delta3: Apparent sampling size of the SLM as seen from the
        sensor plane
        :param delta4: Sampling size of the sensor plane
        :return u3_est, u4_est, idx_converge: Estimated fields of size (Ny,Nx)
        and the convergence indices to check convergence speed
        """
        wvl = self.wavelength
        z3 = self.z
        # parameters
        Nx = y0.shape[1]
        Ny = y0.shape[0]
        Nim = self.Nim
        N_os = self.N_os
        N_iter = self.N_gs
        k = 2 * np.pi / wvl
        xx = np.linspace(0, Nx - 1, Nx, dtype=np.float32) - \
            (Nx / 2) * np.ones(Nx, dtype=np.float32)
        yy = np.linspace(0, Ny - 1, Ny, dtype=np.float32) - \
            (Ny / 2) * np.ones(Ny, dtype=np.float32)
        X, Y = float(delta4x) * np.meshgrid(xx, yy)[0], \
            float(delta4y) * np.meshgrid(xx, yy)[1]
        R = np.sqrt(X ** 2 + Y ** 2)
        Q = np.exp(1j * (k / (2 * z3)) * R ** 2)
        del xx, yy, X, Y, R
        SLM = SLM.repeat(N_os, axis=2)
        SLM = SLM.transpose(2, 0, 1)
        U3 = pyfftw.empty_aligned((Nim, Ny, Nx), dtype=np.complex64)
        y = pyfftw.empty_aligned((Nim, Ny, Nx), dtype=np.complex64)
        with open('fft_wisdom.pickle', 'rb') as f:
            fft_wisdom = pickle.load(f)
        pyfftw.import_wisdom(fft_wisdom)
        fft_obj = pyfftw.builders.fft2(U3, axes=(1, 2),
                                       overwrite_input=True,
                                       threads=multiprocessing.cpu_count()//2,
                                       planner_effort="FFTW_PATIENT")
        ifft_obj = pyfftw.builders.ifft2(y, axes=(1, 2),
                                         overwrite_input=True,
                                         threads=multiprocessing.cpu_count()//2,
                                         planner_effort="FFTW_PATIENT")
        fft_wisdom = pyfftw.export_wisdom()
        with open('fft_wisdom.pickle', 'wb') as f:
            pickle.dump(fft_wisdom, f)
        y0 = y0.transpose(2, 0, 1)
        for ii in range(N_os):
            y_batch = y0[ii, :, :]
            SLM_batch = SLM[ii, :, :]
            U3[ii, :, :] = self.frt_s(y_batch / Q, delta4x, delta4y,
                                      self.wavelength, -z3)*np.conj(SLM_batch)
        u3 = np.mean(U3[0:N_os, :, :], 0)
        del SLM_batch, y_batch
        # Recon run : GS loop
        idx_converge = np.empty(N_iter//5)
        T_run_0 = time.time()
        # fftshift first 
        SLM = np.fft.ifftshift(SLM, axes=(1, 2))
        u3 = np.fft.ifftshift(u3)
        y0 = np.fft.fftshift(y0, axes=(1, 2)) 
        for jj in range(N_iter):
            sys.stdout.flush()
            # on the sensor
            U3 = self.frt_vec_s(SLM * u3, delta3x, delta3y, self.wavelength,
                                z3, fft=fft_obj)
            # convergence index matrix for every 5 iterations
            if jj % 5 == 0:
                idx_converge0 = (1 / np.sqrt(Nx*Ny)) * \
                    np.linalg.norm((np.abs(U3)-y0) * (y0 > 0), axis=(1, 2))
                idx_converge[jj//5] = np.mean(idx_converge0)
                prt = f"\rGS iteration {jj + 1}  (convergence index : {idx_converge[jj//5]})"
                sys.stdout.write(prt)
            U3 = y0 * np.exp(1j * np.angle(U3))  # impose the amplitude
            U3 = self.frt_vec_s(U3, delta4x, delta4y, self.wavelength, -z3,
                                fft=ifft_obj) * np.conj(SLM)
            u3 = np.mean(U3, 0)  # average over batches
            sys.stdout.write(f"\r GS iteration {jj + 1}")
            # exit if the matrix doesn't change much
            if (jj > 1) & (jj % 5 == 0):
                eps = np.abs(idx_converge[jj//5] - idx_converge[jj//5 - 1]) / \
                    idx_converge[jj//5]
                if eps < 1e-3:
                    print('\nConverged. Exit the GS loop ...')
                    idx_converge = idx_converge[0:jj//5]
                    break
        # fftshift
        u3 = np.fft.fftshift(u3)
        # propagate solution to sensor plane
        u4_est = self.frt_s(u3, delta3x, delta3y, self.wavelength, z3) * Q
        T_run = time.time()-T_run_0
        print(f"\n Time spent in the GS loop : {T_run} s")
        return u3, u4_est, idx_converge