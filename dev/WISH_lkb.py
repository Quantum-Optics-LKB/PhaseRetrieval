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
from cupyx.scipy.ndimage import zoom, shift
from cupyx.scipy import fftpack

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

    def crop_center(self, img, cropx, cropy):
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
        phi_m = cp.asnumpy(zoom(M, (shape[0]/M.shape[0], shape[1]/M.shape[1])))
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
                zoom(M, (shape[0] / M.shape[0], shape[1] / M.shape[1])))
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
    def frt(A0: np.ndarray, d1x: float, d1y: float, wv: float, z: float):
        """
        Implements propagation using Fresnel diffraction
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
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
    def frt_gpu(A0: np.ndarray, d1x: float, d1y: float, wv: float, z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using
        CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
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
                    z: float):
        """
        Implements propagation using Fresnel diffraction. Runs on a GPU using
        CuPy with a CUDA backend.
        :param A0: Field to propagate
        :param d1x: Sampling size of the field A0 in the x direction
        :param d1y: Sampling size of the field A0 in the y direction
        :param wv: Wavelength in m
        :param z : Propagation distance in metres
        :return: A0 : Propagated field
        """
        k = 2 * np.pi / wv
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
            A0 = cp.fft.fft2(A0, axes=(0, 1))
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = d1x*d1y * A0
        elif z <= 0:
            A0 = cp.fft.ifftshift(A0, axes=(0, 1))
            A0 = cp.fft.ifft2(A0, axes=(0, 1))
            A0 = cp.fft.fftshift(A0, axes=(0, 1))
            A0 = (Nx*d1x*Ny*d1y) * A0
        A0 = A0 * 1 / (1j * wv * z)
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
        Nx = A0.shape[2]
        Ny = A0.shape[1]
        if z > 0:
            if plan is None:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                A0 = cp.fft.fft2(A0, axes=(1, 2))
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = d1x*d1y * A0
            else:
                with plan:
                    A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                    A0 = cp.fft.fft2(A0, axes=(1, 2))
                    A0 = cp.fft.fftshift(A0, axes=(1, 2))
                    A0 = d1x*d1y * A0
        elif z <= 0:
            if plan is None:
                A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                A0 = cp.fft.ifft2(A0, axes=(1, 2))
                A0 = cp.fft.fftshift(A0, axes=(1, 2))
                A0 = (Nx*d1x*Ny*d1y) * A0
            else:
                with plan:
                    A0 = cp.fft.ifftshift(A0, axes=(1, 2))
                    A0 = cp.fft.ifft2(A0, axes=(1, 2))
                    A0 = cp.fft.fftshift(A0, axes=(1, 2))
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

    def process_SLM(self, slm: np.ndarray, Nx: int, Ny: int, delta3x: float,
                    delta3y: float, type: str):
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
        if slm.dtype == 'uint8':
            slm = slm.astype(float)/256.
        if slm.ndim == 3:
            slm3 = np.empty((Ny, Nx, N_batch))
            # scale SLM slices to the right size
            for i in range(N_batch):
                slm1 = cp.asnumpy(
                        zoom(cp.asarray(slm[:, :, i]),
                             [delta_SLM/delta3y, delta_SLM/delta3x], order=0))
                if slm1.shape[0] > Ny or slm1.shape[1] > Nx:
                    slm1 = self.crop_center(slm1, Nx, Ny)
                    slm3[:, :, i] = slm1
                else:
                    slm1 = np.pad(slm1,
                                  (int(np.ceil((Ny - slm1.shape[0]) / 2)),
                                   int(np.ceil((Nx - slm1.shape[1]) / 2))))
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
            a4 = self.frt_gpu(a31, delta3x, delta3y, self.wavelength, z3)
            w = noise * cp.random.standard_normal((Ny, Nx,), dtype=float)
            ya = cp.abs(a4)**2 + w
            ya[ya < 0] = 0
            # ya = shift(ya, (1*cp.random.standard_normal(1, dtype=float),
            # 1*cp.random.standard_normal(1, dtype=float)))
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
        k = 2 * np.pi / wvl
        xx = cp.linspace(0, Nx - 1, Nx, dtype=cp.float) - (Nx / 2) *\
            cp.ones(Nx, dtype=cp.float)
        yy = cp.linspace(0, Ny - 1, Ny, dtype=cp.float) - (Ny / 2) *\
            cp.ones(Ny, dtype=cp.float)
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
        U3 = cp.empty((Nim, Ny, Nx), dtype=cp.complex64)  # store all U3 gpu
        k = 2 * np.pi / wvl
        xx = cp.linspace(0, Nx - 1, Nx, dtype=cp.float) - \
            (Nx / 2) * cp.ones(Nx, dtype=cp.float)
        yy = cp.linspace(0, Ny - 1, Ny, dtype=cp.float) - \
            (Ny / 2) * cp.ones(Ny, dtype=cp.float)
        X, Y = float(delta4x) * cp.meshgrid(xx, yy)[0], \
            float(delta4y) * cp.meshgrid(xx, yy)[1]
        R = cp.sqrt(X ** 2 + Y ** 2)
        Q = cp.exp(1j * (k / (2 * z3)) * R ** 2)
        del xx, yy, X, Y, R
        SLM = cp.asarray(SLM.repeat(N_os, axis=2))
        y0 = cp.asarray(y0)
        SLM = SLM.transpose(2, 0, 1)
        y0 = y0.transpose(2, 0, 1)
        for ii in range(N_os):
            y0_batch = y0[ii, :, :]
            SLM_batch = y0[ii, :, :]
            U3[ii, :, :] = self.frt_gpu_s(y0_batch / Q, delta4x, delta4y,
                                          self.wavelength, -z3) *\
                cp.conj(SLM_batch)  # y0_batch gpu
        u3 = cp.mean(U3[0:N_os, :, :], 0)
        del SLM_batch, y0_batch
        # Recon run : GS loop
        idx_converge = np.empty(N_iter)
        plan_fft = fftpack.get_fft_plan(U3, axes=(1, 2))
        for jj in range(N_iter):
            sys.stdout.flush()
            # on the sensor
            U3 = self.frt_gpu_vec_s((SLM * u3), delta3x, delta3y,
                                    self.wavelength, z3, plan=plan_fft)
            # convergence index matrix for each batch
            idx_converge0 = (1 / np.sqrt(Nx*Ny)) * \
                cp.linalg.norm((cp.abs(U3)-y0) * (y0 > 0), axis=(1, 2))
            U3 = y0 * cp.exp(1j * cp.angle(U3))  # impose the amplitude
            U3 = self.frt_gpu_vec_s(U3, delta4x, delta4y, self.wavelength,
                                    -z3, plan=plan_fft) * cp.conj(SLM)

            u3 = cp.mean(U3, 0)  # average over batches
            idx_converge[jj] = np.mean(idx_converge0)  # sum over batches
            sys.stdout.write(f"\rGS iteration {jj + 1}")
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")

            # exit if the matrix doesn't change much
            if jj > 1:
                eps = cp.abs(idx_converge[jj] - idx_converge[jj - 1]) / \
                    idx_converge[jj]
                if eps < 5e-5:
                    # if cp.abs(idx_converge[jj]) < 5e-6:
                    # if idx_converge[jj]>idx_converge[jj-1]:
                    print('\nConverged. Exit the GS loop ...')
                    # idx_converge = idx_converge[0:jj]
                    idx_converge = cp.asnumpy(idx_converge[0:jj])
                    break
        # propagate solution to sensor plane
        u4_est = self.frt_gpu_s(u3, delta3x, delta3y, self.wavelength, z3) * Q
        return u3, u4_est, idx_converge
