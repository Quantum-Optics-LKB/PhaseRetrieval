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
    phi0 = np.array(Image.open('phases/smiley_256.bmp'))[:,:,0]
    im = cp.asnumpy(zoom(cp.asarray(im), 1))
    phi0 = cp.asnumpy(zoom(cp.asarray(phi0), 1))
    padding=64
    u40 = np.pad(im.astype(np.float)/256, (padding, padding)) #protection band
    u40 = Sensor.gaussian_profile(u40, 0.5)
    phi0 = np.pad(phi0.astype(np.float)/256, (padding,padding)) #protection band
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
        slm[:,:,1] = Sensor.modulate_binary((1080, 1920), pxsize=1)
        for i in range(1,Sensor.N_mod//2):
            slm[:, :, 2 * i] = Sensor.modulate_binary((1080, 1920), pxsize=1)
            slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
    elif slm_type=='SLM':
        slm = np.ones((1024, 1280, Sensor.N_mod))
        for i in range(1,Sensor.N_mod):
            slm[:,:,i]=Sensor.modulate((1024,1280), pxsize=1)
    if slm_type =='DMD':
        SLM = Sensor.process_SLM(slm, N, delta3, type="amp")
        SLM[np.abs(SLM) > 0.5] = 1 + 1j*0
        SLM[SLM <= 0.5] = 0 + 1j*0
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
        im2=ax2.imshow(np.abs(u30), vmin=0, vmax=1)
        ax1.set_title("SLM modulation pattern")
        ax2.set_title("Back-propagated field")
        cbar1=fig.colorbar(im1, cax=cax1)
        cbar2=fig.colorbar(im2, cax=cax2)
        cbar1.set_label("Phase in rad", rotation=270)
        cbar2.set_label("Intensity", rotation=270)
        plt.show()
    ims = np.mean(np.asarray([Sensor.gen_ims(u30, SLM, z3, delta3, noise) for _ in range(100)]), axis=0)
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
    u3_est, u4_est, idx_converge = Sensor.WISHrun(y0, SLM, delta3, delta4, plot=False)
    T_run=time.time()-T_run_0
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)
    phase_RMS =(1/(2*np.pi*N)) * np.array(
        [np.linalg.norm((np.angle(u40)-np.angle(np.exp(1j*th)*u4_est))*(np.abs(u40) > 0)) for th in
         np.linspace(-np.pi, np.pi, 256)])
    phase_rms = np.min(phase_RMS)
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
    ax5 = fig.add_subplot(236)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    im1=ax1.imshow(np.abs(u40), cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Initial amplitude')
    im2=ax2.imshow(np.angle(u40), cmap='twilight',vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Initial phase')
    im3=ax3.imshow(abs(u4_est)**2, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Amplitude estimation')
    im4=ax4.imshow(np.angle(u4_est), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase estimation')
    ax5.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax5.set_yscale('log')
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    cbar1=fig.colorbar(im1, cax=cax1)
    cbar2=fig.colorbar(im2, cax=cax2)
    cbar3=fig.colorbar(im3, cax=cax3)
    cbar4=fig.colorbar(im4, cax=cax4)
    cbar1.set_label("Intensity", rotation=270)
    cbar2.set_label("Phase in rad", rotation=270)
    cbar3.set_label("Intensity", rotation=270)
    cbar4.set_label("Phase in rad", rotation=270)
    plt.show()
    #for modulation pixel size run
    fig=plt.figure()
    #ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(111)
    #divider1 = make_axes_locatable(ax1)
    #cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    #im1 = ax1.imshow(np.angle(SLM[:, :, Sensor.N_os-1]), vmin=-np.pi, vmax=np.pi, cmap='twilight')
    #ax1.set_title("SLM modulation pattern")
    im2 = ax2.imshow(np.angle(u4_est), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.text(8, 18, f"RMS = {'{:.2f}%'.format(100*phase_rms)}", bbox={'facecolor': 'white', 'pad': 4})
    ax2.set_title('Phase estimation')
    #cbar1 = fig.colorbar(im1, cax=cax1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    #cbar1.set_label("Phase in rad", rotation=270)
    cbar2.set_label("Phase in rad", rotation=270)
    plt.show()
if __name__=="__main__":
    main()