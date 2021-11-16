# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 08/06/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cupy as cp
from cupyx.scipy.ndimage import interpolation, zoom
from WISH_lkb import WISH_Sensor, WISH_Sensor_cpu


def main():
    # start timer
    T0 = time.time()
    # instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4x = Sensor.d_CAM
    delta4y = Sensor.d_CAM
    im = np.array(Image.open('intensities/I0_256_full.bmp'))[:, :, 0]
    # im = np.load('measurements/y0_32.npy')[:,:,0]**2
    phi0 = np.array(Image.open('phases/harambe_256_full.bmp'))[:, :, 0]
    # im = cp.asnumpy(zoom(cp.asarray(im), [7, 7]))
    # phi0 = cp.asnumpy(zoom(cp.asarray(phi0), [7, 7]))
    im = cp.asnumpy(zoom(cp.asarray(im), [4, 4]))
    phi0 = cp.asnumpy(zoom(cp.asarray(phi0), [4, 4]))
    # paddingy = int((2160-(7*256))/2)
    # paddingx = int((3840-(7*256))/2)
    paddingy = int((2048-(4*256))/2)
    paddingx = int((2048-(4*256))/2)
    u40 = np.pad(im.astype(np.float32)/256, ((paddingy, paddingy),
                                             (paddingx, paddingx)))
    u40 = Sensor.gaussian_profile(u40, 0.5)
    phi0 = np.pad(phi0.astype(np.float32)/256, ((paddingy, paddingy),
                                                (paddingx, paddingx)))
    u40 = u40 * (np.exp(1j * phi0 * 2 * np.pi))
    u40 = u40.astype(np.complex64)
    Nx = u40.shape[1]
    Ny = u40.shape[0]
    # Nx = 2048
    # Ny = 2048
    delta3x = wvl * z3 / (Nx * delta4x)
    delta3y = wvl * z3 / (Ny * delta4y)
    u30 = Sensor.u4Tou3(u40, delta4x, delta4y, z3)
    # forward prop to the sensor plane with SLM modulation
    print('Generating simulation data images ...')
    noise = Sensor.noise
    slm_type = 'DMD'
    if slm_type == 'DMD':
        slm = np.ones((1080, 1920, Sensor.N_mod))
        for i in range(0, Sensor.N_mod//2):
            slm[:, :, 2 * i] = 2*Sensor.modulate_binary((1080, 1920), pxsize=1)
            slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
    elif slm_type == 'SLM':
        slm = np.ones((1080, 1920, Sensor.N_mod))
        for i in range(0, Sensor.N_mod):
            slm[:, :, i] = Sensor.modulate((slm.shape[0], slm.shape[1]),
                                           pxsize=8)
    if slm_type == 'DMD':
        SLM = Sensor.process_SLM(slm, Nx, Ny, delta4x, delta4y, type="amp")
        # DMD angle corrections
        th1 = 0.0 * (np.pi/180)
        th2 = -45.0 * (np.pi/180)
        x = (SLM.shape[1]//2 - np.linspace(0, SLM.shape[1]-1, SLM.shape[1]))
        y = (SLM.shape[0]//2 - np.linspace(0, SLM.shape[0]-1, SLM.shape[0]))
        X, Y = np.meshgrid(x, y)
        X1, Y1 = np.cos(th2)*X*Sensor.d_CAM - np.sin(th2)*Y*Sensor.d_CAM, np.sin(th2)*X*Sensor.d_CAM + np.cos(th2)*Y*Sensor.d_CAM
        dmd_angle_corr = np.exp(1j*X1*np.tan(th1)*2*np.pi/Sensor.wavelength)
        alpha = th1
        theta = 12.0 * (np.pi/180)
        theta_1D = np.arctan(np.tan(theta)/np.sqrt(2))
        beta = -alpha +2*theta # reflection angle
        alpha_1D = np.arctan(np.tan(alpha)/np.sqrt(2))
        beta_1D = 2*theta_1D-alpha_1D#np.arctan(np.tan(beta)/np.sqrt(2))
        dmd_blaz_corr = np.exp((X-Y)*complex(0,1)*2*np.pi/Sensor.wavelength*Sensor.d_SLM*(np.sin(alpha_1D)+np.sin(beta_1D)))
        # SLM = np.array([SLM[:, :, i]*dmd_blaz_corr for i in range(SLM.shape[2])]).transpose((1, 2, 0))
        fig, ax = plt.subplots(1, 3)
        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        im0 = ax[0].imshow(np.angle(SLM[:, :, Sensor.N_os+1]), vmin=-np.pi, vmax=np.pi, cmap="twilight", interpolation=None)
        ax[0].set_title("DMD phase correction")
        im1 = ax[1].imshow(np.abs(SLM[:, :, Sensor.N_os+1]), vmin=0, vmax=1, interpolation=None)
        ax[1].set_title("DMD modulation pattern")
        im2 = ax[2].imshow(np.abs(u30), vmin=0, vmax=1, interpolation=None)
        ax[2].set_title("Back propagated field")
        cbar0 = fig.colorbar(im0, cax=cax0)
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar0.set_label("Phase", rotation=270)
        cbar1.set_label("Modulation amplitude", rotation=270)
        cbar2.set_label("Intensity", rotation=270)
        plt.show()
    elif slm_type == 'SLM':
        SLM = Sensor.process_SLM(slm, Nx, Ny, delta4x, delta4y, type="phi")
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        im1 = ax1.imshow(np.angle(SLM[:, :, 1]), vmin=-np.pi, vmax=np.pi,
                         cmap='twilight')
        im2 = ax2.imshow(np.abs(u30))  # , vmin=0, vmax=1)
        ax1.set_title("SLM modulation pattern")
        ax2.set_title("Back-propagated field")
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar1.set_label("Phase in rad", rotation=270)
        cbar2.set_label("Intensity", rotation=270)
        plt.show()
    # u30 *= np.conj(dmd_angle_corr)
    ims = Sensor.gen_ims(u30, SLM, z3, delta3x, delta3y, noise)
    print('\nCaptured images are simulated')
    # reconstruction
    # process the captured image : converting to amplitude and padding if
    # needed
    y0 = Sensor.process_ims(ims, Nx, Ny)
    # np.save("y0.npy", y0)
    # np.save("SLM.npy", SLM)
    fig = plt.figure(0)
    ax1 = plt.gca()
    ax1.set_title("Modulated field at the sensor plane")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.imshow(y0[:, :, y0.shape[2]-1], vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Intensity", rotation=270)
    plt.show()
    # y0 = np.load("y0.npy")
    # SLM = np.load("SLM.npy")
    # Recon initilization
    u3_est, u4_est, idx_converge = Sensor.WISHrun_vec(
                            y0, SLM, delta3x, delta3y, delta4x, delta4y)
    u41 = cp.asarray(u40)
    diff = ((cp.angle(u41) - cp.angle(u4_est))%(2*np.pi)) * (cp.abs(u41) > 0)
    phase_rms = cp.linalg.norm(diff[cp.abs(u41) > 0]-cp.mean(diff[cp.abs(u41) > 0]))/(2*np.pi*(np.sqrt((Nx-2*paddingx)*(Ny-2*paddingy))))
    phase_pv = cp.nanmax(diff[cp.abs(u41) > 0]-cp.mean(diff[cp.abs(u41) > 0]))-cp.nanmin(diff[cp.abs(u41) > 0]-cp.mean(diff[cp.abs(u41) > 0]))
    # phase_RMS = (1/(2*np.pi*(np.sqrt((Nx-2*paddingx)*(Ny-2*paddingy))))) * \
    #     np.asarray([np.linalg.norm((np.angle(u40) - np.angle(np.exp(1j * th) * u4_est)) *
    #                     (np.abs(u40) > 0)) for th in np.linspace(
    #                     -np.pi, np.pi, 512)])
    # phase_rms = cp.asnumpy(cp.min(phase_RMS))
    # phase_rms = np.min(phase_RMS)
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)

    print(f"\n Phase RMS is {'{:.6f} %'.format(100*phase_rms)}. Phase PV is {'{:.6f} rad'.format(phase_pv)}")
    # total time
    T = time.time()-T0
    print(f"\n Total time elapsed : {T} s")
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    divider3 = make_axes_locatable(ax2)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.imshow(np.abs(u40), cmap='viridis', vmin=0, vmax=1, interpolation=None)
    ax1.set_title('Initial amplitude', fontsize=14)
    im2 = ax4.imshow(np.angle(u40), cmap='twilight', vmin=-np.pi, vmax=np.pi, interpolation=None)
    ax4.set_title('Initial phase', fontsize=14)
    im3 = ax2.imshow(np.abs(u4_est), cmap='viridis', vmin=0, vmax=1, interpolation=None)
    ax2.set_title('Amplitude estimation cam', fontsize=14)
    diff_show = cp.asnumpy(diff*(cp.abs(u41) > 0))
    im5 = ax5.imshow(diff_show, cmap='viridis', vmin=np.nanmin(diff_show[np.abs(u40) > 0]),
                     vmax=np.nanmax(diff_show[np.abs(u40) > 0]), interpolation=None)
    # ax5.text(8, 18, f"RMS = {'{:.3f}%'.format(100 * phase_rms)}",
            #  bbox={'facecolor': 'white', 'pad': 4})
    ax5.set_title('Phase difference', fontsize=14)
    # ax3.plot(np.arange(0, len(idx_converge)*5, 5), idx_converge)
    ax3.plot(idx_converge)
    ax3.set_yscale('log')
    ax3.set_title("Convergence curve", fontsize=14)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMS error of the estimated field")
    ax6.plot(diff_show[1024, paddingx:2048-paddingx])
    ax6.set_yscale('linear')
    ax6.set_title("Cut along center", fontsize=14)
    ax6.set_xlabel("Position")
    ax6.set_ylabel("Difference")
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar3 = fig.colorbar(im3, cax=cax3)
    cbar5 = fig.colorbar(im5, cax=cax5)
    cbar1.set_label("Intensity", rotation=270)
    cbar2.set_label("Phase in rad", rotation=270)
    cbar3.set_label("Intensity", rotation=270)
    cbar5.set_label("Phase in rad", rotation=270)
    plt.show()
    # for modulation pixel size run
    # fig = plt.figure()
    # # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(111)
    # # divider1 = make_axes_locatable(ax1)
    # # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    # # im1 = ax1.imshow(np.abs(SLM[:, :, Sensor.N_os]), vmin=0, vmax=1, cmap='gray')
    # # ax1.set_title("DMD modulation pattern")
    # im2 = ax2.imshow(np.angle(u4_est), cmap='twilight', vmin=-np.pi,
    #                  vmax=np.pi)
    # ax2.text(8, 18, f"RMS = {'{:.3f}%'.format(100*phase_rms)}",
    #          bbox={'facecolor': 'white', 'pad': 4})
    # ax2.set_title('Phase estimation')
    # # cbar1 = fig.colorbar(im1, cax=cax1)
    # cbar2 = fig.colorbar(im2, cax=cax2)
    # # cbar1.set_label("Phase in rad", rotation=270)
    # cbar2.set_label("Phase in rad", rotation=270)
    # plt.show()


if __name__ == "__main__":
    main()
