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
import time
import sys
import configparser
import ast
from scipy import signal, interpolate
import cupy, cupyx
import multiprocessing
from scipy.ndimage import gaussian_filter, zoom

class WavefrontSensor:
    def __init__(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path)
        self.size_SLM = float(conf["params"]["size_SLM"])  # size of the SLM window
        self.size = float(conf["params"]["size"])  # size of the SLM window
        self.d_SLM = float(conf["params"]["d_SLM"])
        self.d_CAM = float(conf["params"]["d_CAM"])
        self.wavelength = float(conf["params"]["wavelength"])
        self.z = float(conf["params"]["z"])  # propagation distance
        self.N_gs = int(conf["params"]["N_gs"])  # number of GS iterations
        self.N_mod = int(conf["params"]["N_mod"])  # number of modulation steps
        self.mod_intensity = float(conf["params"]["mod_intensity"])  # modulation intensity
        self.SLM_levels = int(conf["params"]["SLM_levels"])  # number of SLM levels
        self.threshold = float(conf['params']['mask_threshold'])  # intensity threshold for the signal region
        self.elements = []  # list of optical elements
        for element in conf["setup"]:
            self.elements.append(ast.literal_eval(conf['setup'][element]))
    # progress bar
    def update_progress(self, progress):
        barLength = 20  # Modify this to change the length of the progress bar
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
        block = int(round(barLength * progress))
        text = "\rProgress : [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block),
                                                    round(progress * 100, ndigits=1), status)
        sys.stdout.write(text)
        sys.stdout.flush()


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
    def FRT(self, A0, d1, z: float):
        """
        Implements propagation using Fresnel diffraction
        :param I0: Intensity to propagate
        :param phi0: Phase of the field
        :param z : Propagation distance in metre
        :return: I, phi : Propagated field
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

    def phase_retrieval_wish(self, I0: np.ndarray, I_target: list, Phi_m: list, unwrap: bool = False, plot: bool = True, **kwargs):
        """
        Assumes the propagation in the provided setup to retrieve the phase from the intensity at the image plane
        :param I0: Source intensity field
        :param I: Intensity sample fields from which to retrieve the phase
        :param Phi_m : Phase modulations
        :param k: Number of iterations for GS algorithm
        :param unwrap : Phase unwrapping at the end
        :param plot : toggle plots
        :param threshold : Threshold for automatic mask float in [0,1] default is 1e-2
        :return phi, FROB : The calculated phase map using Gerchberg-Saxton algorithm and differences between each iterations
        """
        # for multiprocessing
        def _GS_iterate_mod(self, phi, Phi_m, I_target, Signal_s, k_s, return_dic):
            """
            Iterates the GS loop once (from iteration N-1 to iteration N). The function will store its outputs in the
            specified lists in arguments.
            :param self: Private function of a WavefrontSensor instance
            :param phi: Phase map at the iteration N-1
            :param phi_m: Modulation applied on the SLM
            :param I_target: Target intensity
            :param signal_s: Signal on the SLM at the iteration N-1
            :return: phi, signal_s : the signal in the SLM plane at iteration N
            """
            # submit new phase (mean phase+modulation)
            signal_s = SubPhase(phi + Phi_m[k_s] , Signal_s[k_s])
            #signal_s = SubPhase(phi , Signal_s[k_s])
            # submit source intensity
            signal_s = SubIntensity(I0, signal_s)
            # signal_s = SubPhase(phi, signal_s)
            #signal_s = Forvard(z, signal_s)  # Propagate to the far field
            signal_s = Fresnel(z, signal_s)  # Propagate to the far field
            # interpolate to target size
            signal_s = Interpol(size, h, 0, 0, 0, 1, signal_s)
            I_f_old = np.reshape(Intensity(1, signal_s), (h, w))  # retrieve far field intensity
            signal_s = SubIntensity(I_target[k_s] * self.mask_sr + I_f_old * self.mask_nr,
                                    signal_s)  # Substitute the measured far field into the field only in the signal region
            signal_s = Forvard(-z, signal_s)  # Propagate back to the near field
            # interpolate to source size
            signal_s = Interpol(size, h_0, 0, 0, 0, 1, signal_s)
            signal_s = SubIntensity(I0, signal_s)  # Substitute the measured near field into the field
            pm_s = np.reshape(Phase(signal_s), I0.shape)
            Signal_s[k_s] = signal_s
            return_dic[str(k_s)] = -pm_s + Phi_m[k_s]

        def _GS_iterate_mod_1(self, phi, Phi_m, I_target, k_s, return_dic):
            """
            Iterates the GS loop once (from iteration N-1 to iteration N). The function will store its outputs in the
            specified lists in arguments.
            :param self: Private function of a WavefrontSensor instance
            :param phi: Phase map at the iteration N-1
            :param phi_m: Modulation applied on the SLM
            :param I_target: Target intensity
            :param signal_s: Signal on the SLM at the iteration N-1
            :return: phi, signal_s : the signal in the SLM plane at iteration N
            """
            # submit new phase (mean phase+modulation)
            # submit source intensity
            d_3 = self.wavelength * self.z / (phi.shape[0] * self.d_CAM)  # apparent pitch in the image plane
            A_s = np.sqrt(I0)*np.exp(1j*(phi + Phi_m[k_s]))
            #propagate to far field
            A_f = self.FRT(A_s,d_3, z)
            I_f_old = np.abs(A_f)**2  # retrieve far field intensity
            A_f = np.sqrt(I_target[k_s] * self.mask_sr + I_f_old * self.mask_nr)*np.exp(1j*np.angle(A_f))
                                     # Substitute the measured far field into the field only in the signal region
            A_s = self.FRT(A_f, self.d_CAM, -z)  # Propagate back to the near field
            pm_s = np.angle(A_s)
            return_dic[str(k_s)] = pm_s - Phi_m[k_s]

        k = self.N_gs
        threshold = self.threshold
        z = self.z
        wavelength = self.wavelength
        size = self.size
        size_SLM = self.size_SLM
        h_0, w_0 = I0.shape
        h, w = I_target[0].shape
        self.mask_sr = self.define_mask(I_target[0], plot)
        self.mask_nr = np.ones(self.mask_sr.shape) - self.mask_sr
        T0 = time.time()
        Signal_s = []
        Phi = []
        # initiate the loop  with a flat phase
        phi = np.zeros(I0.shape)
        phi0_sr[np.where(I0 == 0)[0], np.where(I0 == 0)[1]] = 0
        phi0_sr[np.where(I0 > 0)[0], np.where(I0 > 0)[1]] = 1
        # initiate fields in the SLM plane
        for phi_m in Phi_m:
            signal_s = Begin(size, wavelength, h)
            signal_s = SubIntensity(I0, signal_s)
            Signal_s.append(signal_s)
            Phi.append(phi)
        FROB=[]
        for i in range(k):
            T1 = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            Processes=[]
            for i_m in range(self.N_mod):
                p = multiprocessing.Process(target=_GS_iterate_mod, args=[self, phi, Phi_m, I_target, Signal_s, i_m, return_dict])
                p.start()
                Processes.append(p)
            for process in Processes:
                process.join()
            Phi = [return_dict.get(str(_)) for _ in range(self.N_mod)]
            phi_new = np.mean(np.array(Phi), axis=0)
            FROB.append(np.linalg.norm(phi0_sr*(phi_new-phi)))
            phi = phi_new
            T2 = time.time() - T1
            progress = float((i + 1) / k)
            self.update_progress(progress)


        if unwrap:
            phi = PhaseUnwrap(phi)
        phi = np.reshape(phi, (h, w))
        T3 = time.time() - T0
        print(f"Elapsed time : {T3} s")
        return phi, np.array(FROB)


    # modulation
    def modulate(self, phi: np.ndarray):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        x=self.mod_intensity
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
        M =  (np.ones((h, w)) - 2 * np.random.rand(h, w))  # random matrix between [-x*pi and x*pi]
        phi_m = np.kron(M, np.ones((10,10)))
        phi_m = gaussian_filter(phi_m, sigma=4)
        phi_m = np.pi * x * phi_m
        return phi_m
    def modulate_binary(self, phi: np.ndarray):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        x=self.mod_intensity
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
        M = np.array([ np.random.choice(np.array([1,-1]), p=[0.5, 0.5]) for _ in range(h*w) ])# random matrix between [-x*pi and x*pi]
        M = np.reshape(M, (h,w))
        phi_m = np.kron(M, np.ones((10,10)))
        phi_m = gaussian_filter(phi_m, sigma=4)
        phi_m = np.pi * x * phi_m
        return phi_m
    def modulate_ternary(self, phi: np.ndarray):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        x=self.mod_intensity
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
        M = np.array([ np.random.choice(np.array([1,0, -1]), p=[0.25, 0.5, 0.25]) for _ in range(h*w) ])# random matrix between [-x*pi and x*pi]
        M = np.reshape(M, (h,w))
        phi_m = np.kron(M, np.ones((10,10)))
        phi_m = gaussian_filter(phi_m, sigma=4)
        phi_m = np.pi * x * phi_m
        return phi_m
    def modulate_quaternary(self, phi: np.ndarray):
        """
        A function to randomly modulating a phase map without introducing too much high frequency noise
        :param phi: Phase map to be modulated
        :return: phi_m a modulated phase map to multiply to phi
        """
        x=self.mod_intensity
        # generate (N/10)x(N/10) random matrices that will then be upscaled through interpolation
        h, w = int(phi.shape[0] / 10), int(phi.shape[1] / 10)
        M = np.array([ np.random.choice(np.array([1,0.5, -0.5,0]), p=[0.25, 0.25, 0.25, 0.25]) for _ in range(h*w) ])# random matrix between [-x*pi and x*pi]
        M = np.reshape(M, (h,w))
        phi_m = np.kron(M, np.ones((10,10)))
        phi_m = gaussian_filter(phi_m, sigma=4)
        phi_m = np.pi * x * phi_m
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
    def generate_images(self, I0, phi0):
        Phi_m=[]
        I_target = []
        for k in range(self.N_mod):
            # for k in range(int(self.N_mod/2)):
            phi_m = self.modulate(phi0)
            Phi_m.append(phi_m)
            # Phi_m.append(-phi_m)
        T0 = time.time()
        A = Begin(self.size_SLM, self.wavelength, I0.shape[0])
        d_3 = self.wavelength * self.z / (phi0.shape[0] * self.d_CAM)
        N = I0.shape[0]
        for phi_m in Phi_m:
            # define target field
            A = SubIntensity(I0, A)
            A = SubPhase(phi0 + phi_m, A)
            A = Fresnel(self.z, A)
            I_f = np.reshape(Intensity(1, A), I0.shape)
            # phi2 =  np.reshape(Phase(A), I0.shape)
            """
            phi2=phi0 + phi_m
            phi2 =zoom(phi2, d_3/self.d_SLM)
            phi2 = np.pad(phi2,((round((N-phi2.shape[0])/2), round((N-phi2.shape[1])/2))))
            if phi2.shape[0]>N:
                phi2=phi2[0:N-1,:]
            if phi2.shape[1]>N:
                phi2=phi2[:,0:N-1]
            A0 = np.sqrt(I0) * np.exp(1j * (phi2))
            # A0 = np.sqrt(I0)*np.exp(1j*(phi0))
            A = self.FRT(A0, d_3, self.z)
            # A = self.FRT(A, d_CAM,-self.z)
            I = np.abs(A) ** 2
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.set_title("FRT")
            ax2 = fig.add_subplot(122)
            ax2.set_title("Fresnel")
            ax1.imshow(I)
            ax2.imshow(I_f)
            plt.show()
            """
            I_target.append(I_f)
        T = time.time() - T0
        print(f"Took me {T} s to generate the modulation")
        I_target = np.array(I_target)
        return I_target, Phi_m
Sensor=WavefrontSensor('wish_1.conf')
# initiate custom phase and intensity filters emulating the SLM
I0 = np.asarray(Image.open("intensities/I0_500.bmp"))[:, :, 0]  # extract only the first channel
phi0 = np.asarray(Image.open("phases/smiley_500.bmp"))[:,:,0]
I0 = Sensor.gaussian_profile(I0, 0.5) / np.max(I0)
phi0 = phi0 / np.max(phi0)
# signal region for the phase
phi0_sr = np.ones(phi0.shape)  # signal region
phi0_sr[np.where(I0 == 0)[0], np.where(I0 == 0)[1]] = 0
phi0_sr[np.where(I0 > 0)[0], np.where(I0 > 0)[1]] = 1
# conversion to rad
#phi0 = 2 * np.pi * (phi0 - 0.5 * np.ones(phi0.shape)) * phi0_sr
phi0 = 2 * np.pi * phi0
Phi_m = []
I_target = []
I_target, Phi_m=Sensor.generate_images(I0, phi0)
I = np.mean(I_target, axis=0)
phi, FROB = Sensor.phase_retrieval_wish(I0, I_target, Phi_m, plot=False)
# compute RMS

#RMS =min([(1 / (2 * np.pi)) * np.sqrt(np.mean(phi0_sr * (phi0 - (phi+a*np.ones(phi.shape))) ** 2)) for a in np.linspace(-np.pi, np.pi, Sensor.SLM_levels)])
RMS =(1 / (2 * np.pi)) * np.sqrt(np.mean(phi0_sr * (phi0 - phi) ** 2))
#FROB =min([ np.linalg.norm(phi0-(phi+a*np.ones(phi.shape))) for a in np.linspace(-np.pi, np.pi, Sensor.SLM_levels)])
frob = np.linalg.norm(phi0_sr*(phi0-phi))
corr = np.corrcoef((phi0_sr*phi).flat, (phi0_sr*phi0).flat)[0, 1]
#T0=time.time()
#phi_fr_cpu = np.fft.fft2(phi)
#T=time.time()-T0
#print(f'Took me {T} second on the CPU')
#phi_gpu = cupy.asarray(phi)
#T0=time.time()
#phi_ft_gpu = cupy.fft.fft2(phi_gpu)
#T=time.time()-T0
#print(f'Took me {T} second on the GPU')

print(f"RMS of the recovered phase is : {RMS}")
print(f'Frobenius norm of the error is : {frob}')
print(f'Correlation coefficient is : {corr}')

fig = plt.figure()
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(336)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes('right', size='5%', pad=0.05)
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes('right', size='5%', pad=0.05)
im1 = ax1.imshow(I0, vmin=0, vmax=1)
im2 = ax2.imshow(phi0, vmin=-np.pi, vmax=np.pi)
im3 = ax3.imshow(I, vmin=0, vmax=1)
im4 = ax4.imshow(phi, vmin=-np.pi, vmax=np.pi)
ax1.set_title("Initial intensity")
ax2.set_title("Initial phase")
ax3.set_title("Mean propagated intensity")
ax4.set_title("Mean retrieved phase")
ax4.text(8, 18, f"RMS = {round(RMS, ndigits=3)}", bbox={'facecolor': 'white', 'pad': 3})
ax5.plot(FROB)
ax5.set_title("Difference between each iteration")
ax5.set_ylabel("Difference in Frobenius norm")
ax5.set_xlabel("Iteration")
fig.colorbar(im1, cax=cax1)
fig.colorbar(im2, cax=cax2)
fig.colorbar(im3, cax=cax3)
fig.colorbar(im4, cax=cax4)
plt.show()
