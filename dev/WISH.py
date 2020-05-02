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
from scipy import signal, interpolate, io
import cupy as cp
import multiprocessing
from scipy.ndimage import gaussian_filter, zoom

class WISH_Sensor:
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
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            im1=ax1.imshow(I, cmap="viridis")
            ax1.set_title("Intensity")
            im2=ax2.imshow(mask_sr, cmap="viridis")
            ax2.set_title(f"Signal region (Threshold = {threshold})")
            scat = ax2.scatter(non_zero[0][R_max], non_zero[1][R_max], color='r')
            scat.set_label('Threshold point')
            ax2.legend()
            fig.colorbar(im1, cax=cax1)
            fig.colorbar(im2, cax=cax2)
            plt.show()
        return mask_sr
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
    def frt(self, A0, d1, z: float):
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
    def frt_gpu(self, A0, d1, z: float):
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
    def frt_gpu_s(self, A0, d1, z: float):
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
        D = 1 /(1j*wv*abs(z))
        if z >=0:
            A =D * (d1**2) * cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(A0), norm='ortho'))
        elif z<0:
            A =D * ((N*d1) ** 2) * cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A0), norm='ortho'))

        return A
    def u4Tou3(self, u4, delta4, z3):
        u3 = self.frt(u4, delta4, -z3);
        return u3
    def gen_ims(self, u3, z3, delta3, Nim, noise):
        # generate ims on the sensor plane
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

        slm = np.array(io.loadmat('/home/tangui/Documents/LKB/WISH/slm60_resize10.mat')['slm'])
        if slm.dtype=='uint8':
            slm = slm.astype(float)/256
        ims = np.zeros((N, N, Nim), dtype=float)

        for i in range(Nim):
            sys.stdout.write(f"\rGenerating image {i+1} out of {Nim} ...")
            sys.stdout.flush()
            slm0 = slm[:, 421: 1500, i]
            slm1 = zoom(slm0, delta_SLM / delta3)
            slm1 = np.pad(slm1, (round((N - slm1.shape[0])/ 2), round((N - slm1.shape[1]) / 2)))
            if slm1.shape[0] > N:
                slm1 = slm1[0:N, :]
            if slm1.shape[1] > N:
                slm1 = slm1[:, 0:N]
            a31 = u3 * A_SLM * np.exp(1j * slm1 * 2 * np.pi)
            a4 = self.frt(a31, delta3, z3)
            w = noise * np.random.rand(N, N)
            ya = np.abs(a4)**2 + w
            ya[ya<0]=0
            ims[:,:, i] = ya
        return ims


    def process_SLM(self, slm, N, Nim, delta3):
        #Scale the SLM to the correct size
        delta_SLM = self.d_SLM
        if slm.dtype == 'uint8':
            slm = slm.astype(float)/256
        slm2 = slm[:, 421: 1501, 0:Nim] #takes a 1080x1080 square of the SLM
        slm3 = np.empty((N,N,Nim))
        #could replace with my modulate function
        #scale SLM slices to the right size
        for i in range(Nim):
            slm1 = zoom(slm2[:,:,i], delta_SLM / delta3)
            slm1 = np.pad(slm1, (round((N - slm1.shape[0]) / 2), round((N - slm1.shape[1]) / 2)))
            if slm1.shape[0] > N:
                slm3[:,:,i] = slm1[0:N, :]
            if slm1.shape[1] > N:
                slm3[:,:,i] = slm1[:, 0:N]

        SLM = np.exp(1j * 2 * np.pi * slm3).astype(np.complex64)
        return SLM
    def process_ims(self, ims, N):
        y0 = np.real(np.sqrt(ims)); # change from intensity to magnitude
        y0 = np.pad(y0, (round((N - y0.shape[0]) / 2), round((N - y0.shape[1]) / 2)))
        if y0.shape[0] > N:
            y0=y0[0:N,0:N,:]
        return y0
    def WISHrun(self, y0, SLM, delta3, delta4, N_os, N_iter, N_batch, plot=True):
        wvl = self.wavelength
        z3 = self.z
        ## parameters
        N = y0.shape[0]
        k = 2 * np.pi / wvl
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
        plt.imshow(cp.asnumpy(cp.abs(u3))) #TODO
        plt.show()

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
                idx_converge0[idx_batch] = cp.asnumpy(cp.mean(cp.mean(cp.mean(y0_batch,1),0)/cp.sum(cp.sum(cp.abs(cp.abs(u4)-y0_batch),1),0)))
                #idx_converge0[idx_batch] = np.linalg.norm(y0_batch)/np.linalg.norm(np.abs(u4)-y0_batch)
                # convergence index matrix for each batch
                del SLM_batch, y0_batch
            u3 = (u3_collect / N_batch) # average over batches
            idx_converge[jj] = np.mean(idx_converge0) # sum over batches
            sys.stdout.write(f"  (convergence index : {idx_converge[jj]})")
            #u4_est = self.frt(u3, delta3, z3)
            u4_est = cp.asnumpy(self.frt_gpu(u3, delta3, z3))

            if jj % 10 == 0 and plot:
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
            #if jj > 1:
            #    if abs(idx_converge[jj] - idx_converge[jj - 1]) / idx_converge[jj] < 1e-4:
            #        print('\nConverged. Exit the GS loop ...')
            #        break
        return u4_est



#WISH routine with the resolution chart
def main():
    ##load parameters and field
    #instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    im = np.array(Image.open('intensities/resChart.bmp'))[:,:,0]
    ## load('I_final.mat')
    u40 = np.pad(im.astype(np.float), (256,256))
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    N = u40.shape[0]
    delta3 = wvl * z3 / (N * delta4)
    u30 = Sensor.u4Tou3(u40, delta4, z3)
    ## forward prop to the sensor plane with SLM modulation
    print('Generating simulation data images ...')
    noise = 0.01
    Nim = 4
    ims = Sensor.gen_ims(u30, z3, delta3, Nim, noise)
    print('\nCaptured images are simulated')
    #clear u30, u40 for memory economy
    del u30
    ## reconstruction
    # pre - process the data
    # for the SLM : correct scaling
    slm = np.array(io.loadmat('/home/tangui/Documents/LKB/WISH/slm60_resize10.mat')['slm'])
    SLM = Sensor.process_SLM(slm, N, Nim, delta3)
    #process the captured image : converting to amplitude and padding if needed
    y0 = Sensor.process_ims(ims, N)
    ##Recon initilization
    N_os = 2 # number of images per batch
    if Nim < N_os:
        N_os = Nim
    N_iter = Sensor.N_gs  # number of GS iterations
    N_batch = int(Nim / N_os)  # number of batches
    u4_est = Sensor.WISHrun(y0, SLM, delta3, delta4, N_os, N_iter, N_batch, plot=True)


    fig=plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    im1=ax1.imshow(np.abs(u40)**2, cmap='viridis')
    ax1.set_title('Amplitude GT')
    im2=ax2.imshow(np.angle(u40), cmap='viridis')
    ax2.set_title('Phase GT')
    im3=ax3.imshow(abs(u4_est), cmap='viridis')
    ax3.set_title('Amplitude estimation')
    im4=ax4.imshow(np.angle(u4_est), cmap='viridis')
    ax4.set_title('Phase estimation')
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    plt.show()
if __name__=="__main__":
    main()
