# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 08/06/2020
"""
# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 28/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cupy as cp
from cupyx.scipy.ndimage import zoom
from WISH_lkb import WISH_Sensor

#WISH routine

def main():
    #start timer
    T0 = time.time()
    #instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    #I0 = np.array(Image.open('intensities/harambe_512_full.bmp'))[:,:,0]
    #I0 = I0.astype(float)/256
    #I0 = np.pad(I0.astype(np.float) / 256, (256, 256))  # protection band
    im = np.array(Image.open('intensities/I0_256_full.bmp'))[:,:,0]
    #im = np.load('measurements/y0_32.npy')[:,:,0]**2
    phi0 = np.array(Image.open('phases/harambe_256_full.bmp'))[:,:,0]
    im = cp.asnumpy(zoom(cp.asarray(im), 8))
    phi0 = cp.asnumpy(zoom(cp.asarray(phi0), 8))
    padding=32
    u40 = np.pad(im.astype(np.float32)/256, (padding, padding)) #protection band
    #u40 = np.pad(im.astype(np.float32), (padding, padding)) #protection band
    u40 = Sensor.gaussian_profile(u40, 0.5)
    phi0 = np.pad(phi0.astype(np.float32)/256, (padding,padding)) #protection band
    u40 = u40 * (np.exp(1j * phi0 * 2 * np.pi))
    u40=u40.astype(np.complex64)
    N = u40.shape[0]
    delta3 = wvl * z3 / (N * delta4)
    u30 = Sensor.u4Tou3(u40, delta4, z3)
    ## forward prop to the sensor plane with SLM modulation
    print('Generating simulation data images ...')
    noise = Sensor.noise
    slm_type = 'SLM'
    if slm_type=='DMD':
        slm = np.ones((1080, 1920, Sensor.N_mod))
        slm[:,:,1] = Sensor.modulate_binary((1080, 1920), pxsize=2)
        for i in range(1,Sensor.N_mod//2):
            slm[:, :, 2 * i] = Sensor.modulate((1080, 1920), pxsize=2)
            slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
    elif slm_type=='SLM':
        slm = np.ones((1152, 1920, Sensor.N_mod))
        for i in range(0, Sensor.N_mod):
            slm[:,:,i]=Sensor.modulate((slm.shape[0], slm.shape[1]), pxsize=1)
    if slm_type =='DMD':
        SLM = Sensor.process_SLM(slm, N, delta3, type="amp")
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        im1=ax1.imshow(np.abs(SLM[:, :, Sensor.N_os+1]), vmin=0, vmax=1)
        ax1.set_title("DMD modulation pattern")
        im2=ax2.imshow(np.abs(u30), vmin=0, vmax=1)
        ax2.set_title("Back propagated field")
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar1.set_label("Phase in rad", rotation=270)
        cbar2.set_label("Intensity", rotation=270)
        plt.show()
    elif slm_type == 'SLM':
        SLM = Sensor.process_SLM(slm, N, delta3, type="phi")
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        im1=ax1.imshow(np.angle(SLM[:,:,1]), vmin=-np.pi, vmax = np.pi, cmap='twilight')
        #im2=ax2.imshow(np.abs(u30), vmin=0, vmax=1)
        im2=ax2.imshow(np.abs(u30)) #, vmin=0, vmax=1)
        ax1.set_title("SLM modulation pattern")
        ax2.set_title("Back-propagated field")
        cbar1=fig.colorbar(im1, cax=cax1)
        cbar2=fig.colorbar(im2, cax=cax2)
        cbar1.set_label("Phase in rad", rotation=270)
        cbar2.set_label("Intensity", rotation=270)
        plt.show()
    ims = Sensor.gen_ims(u30, SLM, z3, delta3, noise)
    print('\nCaptured images are simulated')
    #reconstruction
    #process the captured image : converting to amplitude and padding if needed
    y0 = Sensor.process_ims(ims, N)

    fig=plt.figure(0)
    ax1=plt.gca()
    ax1.set_title("Modulated field at the sensor plane")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im1=ax1.imshow(y0[:,:,y0.shape[2]-1], vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Intensity", rotation=270)
    plt.show()

    ##Recon initilization
    T_run_0=time.time()
    u3_est, u4_est, idx_converge = Sensor.WISHrun_vec(y0, SLM, delta3, delta4)
    T_run=time.time()-T_run_0
    u41=cp.asarray(u40)
    phase_RMS = (1 / (2 * np.pi * (N - 2 * padding))) * cp.asarray(
        [cp.linalg.norm((cp.angle(u41) - cp.angle(cp.exp(1j * th) * u4_est)) * (cp.abs(u41) > 0)) for th in
         cp.linspace(-np.pi, np.pi, 512)])
    phase_rms = cp.asnumpy(cp.min(phase_RMS))
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)


    print(f"\n Phase RMS is : {phase_rms}")
    #total time
    T= time.time()-T0
    print(f"\n Time spent in the GS loop : {T_run} s")
    print(f"\n Total time elapsed : {T} s")
    fig=plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    divider3 = make_axes_locatable(ax2)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    im1=ax1.imshow(np.abs(u40), cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Initial amplitude', fontsize=20)
    im2=ax4.imshow(np.angle(u40), cmap='twilight',vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Initial phase', fontsize=20)
    im3=ax2.imshow(np.abs(u3_est), cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Amplitude estimation', fontsize=20)
    im5=ax5.imshow(np.angle(u4_est), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax5.text(8, 18, f"RMS = {'{:.3f}%'.format(100 * phase_rms)}", bbox={'facecolor': 'white', 'pad': 4})
    ax5.set_title('Phase estimation', fontsize=20)
    ax3.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax3.set_yscale('log')
    ax3.set_title("Convergence curve", fontsize=20)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMS error of the estimated field")
    cbar1=fig.colorbar(im1, cax=cax1)
    cbar2=fig.colorbar(im2, cax=cax2)
    cbar3=fig.colorbar(im3, cax=cax3)
    cbar5=fig.colorbar(im5, cax=cax5)
    cbar1.set_label("Intensity", rotation=270)
    cbar2.set_label("Phase in rad", rotation=270)
    cbar3.set_label("Intensity", rotation=270)
    cbar5.set_label("Phase in rad", rotation=270)
    plt.show()
    #for modulation pixel size run
    fig=plt.figure()
    #ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(111)
    #divider1 = make_axes_locatable(ax1)
    #cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    #im1 = ax1.imshow(np.abs(SLM[:, :, Sensor.N_os]), vmin=0, vmax=1, cmap='gray')
    #ax1.set_title("DMD modulation pattern")
    im2 = ax2.imshow(np.angle(u4_est), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.text(8, 18, f"RMS = {'{:.3f}%'.format(100*phase_rms)}", bbox={'facecolor': 'white', 'pad': 4})
    ax2.set_title('Phase estimation')
    #cbar1 = fig.colorbar(im1, cax=cax1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    #cbar1.set_label("Phase in rad", rotation=270)
    cbar2.set_label("Phase in rad", rotation=270)
    plt.show()
if __name__=="__main__":
    main()