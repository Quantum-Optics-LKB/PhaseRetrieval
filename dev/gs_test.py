import numpy as np
import pyfftw
import matplotlib.pyplot as plt
import multiprocessing

Nx = 1080
Ny = 1920
wv = 795e-9
A = pyfftw.empty_aligned((Ny, Nx), dtype=np.complex64)
k = 2 * np.pi / wv
x = np.linspace(0, Nx - 1, Nx, dtype=np.float32) -\
    (Nx / 2) * np.ones(Nx, dtype=np.float32)
y = np.linspace(0, Ny - 1, Ny, dtype=np.float32) -\
    (Ny / 2) * np.ones(Ny, dtype=np.float32)
d1x = 8.0e-6
d1y = 8.0e-6
X1, Y1 = d1x * np.meshgrid(x, y)[0], d1y * np.meshgrid(x, y)[1]
R1 = np.sqrt(X1 ** 2 + Y1 ** 2)
def GS(E_target, E, z, N_it):
    d2x = wv * abs(z) / (Nx * d1x)
    d2y = wv * abs(z) / (Ny * d1y)
    X2, Y2 = d2x * np.meshgrid(x, y)[0], d2y * np.meshgrid(x, y)[1]
    R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
    fft_obj = pyfftw.builders.fft2(E, axes=(0, 1),
                                        overwrite_input=True,
                                        threads=multiprocessing.cpu_count()//2,
                                        planner_effort="FFTW_PATIENT")
    ifft_obj = pyfftw.builders.ifft2(E_target, axes=(0, 1),
                                        overwrite_input=True,
                                        threads=multiprocessing.cpu_count()//2,
                                        planner_effort="FFTW_PATIENT")                
    def frt(A0: np.ndarray, z):
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
            if z > 0:
                fft_obj(A0, A0)
                A0 *= d1x*d1y 
            elif z <= 0:
                ifft_obj(A0, A0)
                A0 *= (Nx*d1x*Ny*d1y) 
            A0 *= 1 / (1j * wv * z)
            return A0
        E *= np.exp(1j * (k / (2 * z)) * R1 ** 2)
        E = np.fft.fftshift(E)
        E_target = np.fft.fftshift(E_target)
        for i in range(N_it):
            E = frt(E, z)
            E = np.abs(E_target) * np.exp(1j*np.angle(E))
            E[:] = frt(E, -z)
        E *= np.exp(1j * (k / (2 * z)) * R2 ** 2)
