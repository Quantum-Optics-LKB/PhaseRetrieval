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
import pyfftw
import mkl_fft
import multiprocessing
import time
import pickle

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


class WISH_Sensor:
    def __init__(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path)
        self.d_SLM = float(conf["params"]["d_SLM"])
        self.d_CAM = float(conf["params"]["d_CAM"])
        self.wavelength = float(conf["params"]["wavelength"])
        # refractive index
        self.n = float(conf["params"]["n"])
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
    def frt(A0: np.ndarray, d1x: float, d1y: float, wv: float, n: float,
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
        k = n*2*np.pi / wv
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
        D = n/(1j*wv*z)
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
    def frt_gpu(A0: np.ndarray, d1x: float, d1y: float, wv: float, n: float,
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
        k = n * 2 * np.pi / wv
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
        A0 = A0 * n / (1j * wv * z)
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
        A0 = A0 * n / (1j * wv * z)
        return A0

    @staticmethod
    def frt_gpu_s(A0: np.ndarray, d1x: float, d1y: float, wv: float, n: float,
                  z: float):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a
        GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: Refractive index
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        Nx = A0.shape[1]
        Ny = A0.shape[0]
        if z > 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = fftsc.fft2(A0, axes=(0, 1), overwrite_x=True)
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = fftsc.ifft2(A0, axes=(0, 1), overwrite_x=True)
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * n / (1j * wv * z)
        return A0

    @staticmethod
    def frt_gpu_vec_s(A0: np.ndarray, d1x: float, d1y: float, wv: float,
                      n: float, z: float, plan=None):
        """
        Simplified Fresnel propagation optimized for GPU computing. Runs on a
        GPU using CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param n: Refractive index
        :param z : Propagation distance in metres
        :return: A : Propagated field
        """
        Nx = A0.shape[2]
        Ny = A0.shape[1]
        if z > 0:
            if plan is None:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                A0 = fftsc.fft2(A0, axes=(1, 2), overwrite_x=True)
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = d1x*d1y * A0
            else:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                plan.fft(A0, A0, cp.cuda.cufft.CUFFT_FORWARD)
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = d1x*d1y * A0
        elif z <= 0:
            if plan is None:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                A0 = fftsc.ifft2(A0, axes=(1, 2), overwrite_x=True)
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = (Nx*d1x*Ny*d1y) * A0
            else:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                plan.fft(A0, A0, cp.cuda.cufft.CUFFT_INVERSE)
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = d1x*d1y * A0
        A0 = A0 * n / (1j * wv * z)
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
        u3 = self.frt(u4, delta4x, delta4y, self.wavelength, self.n, -z3)
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
            slm = slm.astype(float)/256.
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
                SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
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
            a31 = cp.asarray(a31)  # put the field in the GPU
            a4 = self.frt_gpu(a31, delta3x, delta3y, self.wavelength, self.n,
                              z3)
            w = noise * cp.random.standard_normal((Ny, Nx,), dtype=float)
            ya = cp.abs(a4)**2 + w
            ya[ya < 0] = 0
            # ya = shift_cp(ya, (1*cp.random.standard_normal(1, dtype=float),
            #               1*cp.random.standard_normal(1, dtype=float)))
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
                                                self.wavelength, self.n, -z3)
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
                                                 delta3y, self.wavelength,
                                                 self.n, z3)
                    # impose the amplitude
                    y[:, :, _] = y0_batch[:, :, _] *\
                        cp.exp(1j * cp.angle(u4[:, :, _]))
                    # [:,:,_] = u4[:,:,_]
                    # y[i_mask:j_mask,i_mask:j_mask,_] = y0_batch[
                    # i_mask:j_mask,i_mask:j_mask,_] \
                    # *cp.exp(1j * cp.angle(u4[i_mask:j_mask,i_mask:j_mask,_]))
                    u3_batch[:, :, _] = self.frt_gpu_s(
                        y[:, :, _], delta4x, delta4y, self.wavelength, self.n, -z3) *\
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
        u4_est = self.frt_gpu_s(u3, delta3x, delta3y, self.wavelength, self.n,
                                z3) * Q
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
        k = self.n * 2 * np.pi / wvl
        xx = cp.linspace(0, Nx - 1, Nx, dtype=cp.float32) - \
            (Nx / 2) * cp.ones(Nx, dtype=cp.float32)
        yy = cp.linspace(0, Ny - 1, Ny, dtype=cp.float32) - \
            (Ny / 2) * cp.ones(Ny, dtype=cp.float32)
        X, Y = float(delta4x) * cp.meshgrid(xx, yy)[0], \
            float(delta4y) * cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j * (k / (2 * z3)) * R ** 2)
        del xx, yy, X, Y, R
        U3 = cp.empty((Nim, Ny, Nx), dtype=cp.complex64)
        SLM = cp.asarray(SLM.repeat(N_os, axis=2), dtype=cp.complex64)
        y0 = cp.asarray(y0, dtype=cp.complex64)
        SLM = SLM.transpose(2, 0, 1)
        y0 = y0.transpose(2, 0, 1)
        for ii in range(N_os):
            y0_batch = y0[ii, :, :]
            SLM_batch = y0[ii, :, :]
            U3[ii, :, :] = self.frt_gpu_s(y0_batch / Q, delta4x, delta4y,
                                          self.wavelength, self.n, -z3) *\
                cp.conj(SLM_batch)  # y0_batch gpu
        u3 = cp.mean(U3[0:N_os, :, :], 0)
        del SLM_batch, y0_batch
        # GS loop
        idx_converge = np.empty(N_iter//5)
        shape = U3.shape
        out_dtype = cp.fft._fft._output_dtype(U3.dtype, 'C2C')
        fft_type = cp.fft._fft._convert_fft_type(out_dtype, 'C2C')
        plan_fft = cp.cuda.cufft.PlanNd(shape[1:], shape[1:], 1,
                                        shape[1]*shape[2], shape[1:], 1,
                                        shape[1]*shape[2], fft_type, shape[0],
                                        order='C', last_axis=-1,
                                        last_size=None)
        T_run_0 = time.time()
        for jj in range(N_iter):
            sys.stdout.flush()
            # on the sensor
            U3 = self.frt_gpu_vec_s(SLM * u3, delta3x, delta3y,
                                    self.wavelength, self.n, z3, plan=plan_fft)
            # convergence index matrix for every 5 iterations
            if jj % 5 == 0:
                idx_converge0 = (1 / np.sqrt(Nx*Ny)) * \
                    cp.linalg.norm((cp.abs(U3)-y0) *
                                   (y0 > 0), axis=(1, 2))
                idx_converge[jj//5] = cp.mean(idx_converge0)
                prt = f"\rGS iteration {jj + 1}  (convergence index : {idx_converge[jj//5]})"
                sys.stdout.write(prt)
            U3 = y0 * cp.exp(1j * cp.angle(U3))  # impose the amplitude
            U3 = self.frt_gpu_vec_s(U3, delta4x, delta4y, self.wavelength,
                                    self.n, -z3, plan=plan_fft) * cp.conj(SLM)
            u3 = cp.mean(U3, 0)  # average over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")

            # exit if the matrix doesn't change much
            if (jj > 1) & (jj % 5 == 0):
                eps = cp.abs(idx_converge[jj//5] - idx_converge[jj//5 - 1]) / \
                    idx_converge[jj//5]
                if eps < 1e-3:
                    # if cp.abs(idx_converge[jj]) < 5e-6:
                    # if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    # idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj//5])
                    break
            if jj == N_iter-1:
                print('\nMax iteration number reached. Exit ...')
        # propagate solution to sensor plane
        u4_est = self.frt_gpu_s(u3, delta3x, delta3y, self.wavelength, self.n,
                                z3) * Q
        T_run = time.time()-T_run_0
        print(f"\n Time spent in the GS loop : {T_run} s")
        return u3, u4_est, idx_converge


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
    def frt(A0: np.ndarray, d1x: float, d1y: float, wv: float, n: float,
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
        k = n * 2 * np.pi / wv
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
    def frt_vec(A0: np.ndarray, d1x: float, d1y: float, wv: float, n: float,
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
                A0 = np.fft.ifftshift(A0, axes=(1, 2))
                A0 = mkl_fft.fft2(A0, axes=(1, 2), overwrite_x=True)
                A0 = np.fft.fftshift(A0, axes=(1, 2))
            else:
                A0 = np.fft.ifftshift(A0, axes=(1, 2))
                A0 = fft(A0)
                A0 = np.fft.fftshift(A0, axes=(1, 2))
            A0 = d1x*d1y * A0

        elif z <= 0:
            if fft is None:
                A0 = np.fft.ifftshift(A0, axes=(1, 2))
                A0 = mkl_fft.ifft2(A0, axes=(1, 2), overwrite_x=True)
                A0 = np.fft.fftshift(A0, axes=(1, 2))
            else:
                A0 = np.fft.ifftshift(A0, axes=(1, 2))
                A0 = fft(A0)
                A0 = np.fft.fftshift(A0, axes=(1, 2))
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
        u3 = self.frt(u4, delta4x, delta4y, self.wavelength, self.n, -z3)
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
        k = self.n * 2 * np.pi / wvl
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
                                       threads=multiprocessing.cpu_count(),
                                       planner_effort="FFTW_PATIENT")
        ifft_obj = pyfftw.builders.ifft2(y, axes=(1, 2),
                                         overwrite_input=True,
                                         threads=multiprocessing.cpu_count(),
                                         planner_effort="FFTW_PATIENT")
        fft_wisdom = pyfftw.export_wisdom()
        with open('fft_wisdom.pickle', 'wb') as f:
            pickle.dump(fft_wisdom, f)
        y = y0.transpose(2, 0, 1)
        for ii in range(N_os):
            y_batch = y[ii, :, :]
            SLM_batch = y[ii, :, :]
            U3[ii, :, :] = self.frt_s(y_batch / Q, delta4x, delta4y,
                                      self.wavelength, -z3)*np.conj(SLM_batch)
        u3 = np.mean(U3[0:N_os, :, :], 0)
        del SLM_batch, y_batch
        # Recon run : GS loop
        idx_converge = np.empty(N_iter//5)
        T_run_0 = time.time()
        for jj in range(N_iter):
            sys.stdout.flush()
            # on the sensor
            U3 = self.frt_vec_s((SLM * u3), delta3x, delta3y, self.wavelength,
                                z3, fft=fft_obj)
            # convergence index matrix for every 5 iterations
            if jj % 5 == 0:
                idx_converge0 = (1 / np.sqrt(Nx*Ny)) * \
                    np.linalg.norm((np.abs(U3)-y) * (y > 0), axis=(1, 2))
                idx_converge[jj//5] = np.mean(idx_converge0)
                prt = f"  (convergence index : {idx_converge[jj//5]})"
                sys.stdout.write(prt)
            U3 = y * np.exp(1j * np.angle(U3))  # impose the amplitude
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
        # propagate solution to sensor plane
        u4_est = self.frt_s(u3, delta3x, delta3y, self.wavelength, z3) * Q
        T_run = time.time()-T_run_0
        print(f"\n Time spent in the GS loop : {T_run} s")
        return u3, u4_est, idx_converge
