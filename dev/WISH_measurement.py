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
import sys
import configparser
import cupy as cp
from scipy.ndimage import zoom
from WISH_lkb import WISH_Sensor
import slmpy
import cv2
import EasyPySpin

#WISH routine
def alignment(frame):
    frame_blurred = cv2.blur(frame, (12, 12))
    ret1, thresh = cv2.threshold(frame, 70, 255, 0)
    thresh_blurred = cv2.blur(thresh, (12, 12))
    M = cv2.moments(thresh_blurred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    T = np.float32([[1, 0, int(frame.shape[1]/2) - cX], [0, 1, int(frame.shape[0]/2) - cY]])
    frame_s = cv2.warpAffine(frame, T, frame.shape)
    return frame_s
def main():
    #start timer
    T0 = time.time()
    #instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    '''
    #I0 = np.array(Image.open('intensities/harambe_512_full.bmp'))[:,:,0]
    #I0 = I0.astype(float)/256
    #I0 = np.pad(I0.astype(np.float) / 256, (256, 256))  # protection band
    im = np.array(Image.open('intensities/I0_256_full.bmp'))[:,:,0]
    phi0 = np.array(Image.open('phases/smiley_256.bmp'))[:,:,0]
    im = cp.asnumpy(zoom(cp.asarray(im), 2))
    phi0 = cp.asnumpy(zoom(cp.asarray(phi0), 2))
    u40 = np.pad(im.astype(np.float)/256, (768, 768)) #protection band
    u40 = Sensor.gaussian_profile(u40, 0.5)
    phi0 = np.pad(phi0.astype(np.float)/256, (768,768)) #protection band
    u40 = u40 * (np.exp(1j * phi0 * 2 * np.pi))
    u40=u40.astype(np.complex64)
    N = u40.shape[0]
    delta3 = wvl * z3 / (N * delta4)
    u30 = Sensor.u4Tou3(u40, delta4, z3)
    ## forward prop to the sensor plane with SLM modulation
    print('Generating simulation data images ...')
    noise = Sensor.noise
    '''
    slm = np.zeros((1080, 1920,Sensor.N_mod))
    slm_type = 'SLM'
    #Setting up the camera for acquisition
    Cam = EasyPySpin.VideoCapture(0) #by default camera 0 is the laptop webcam

    N = int(Cam.get(cv2.CAP_PROP_FRAME_WIDTH)*1)
    delta3 = wvl * z3 / (N * delta4)
    ims = np.zeros((N,N,Sensor.Nim))
    if slm_type=='DMD':
        slm = np.ones((1080, 1920, Sensor.N_mod))
        slm_display = slmpy.SLMdisplay()
        slm_display.updateArray(slm[:, :, 0])
        print(f"Displaying 1 st SLM pattern")
        for obs in range(Sensor.N_os):
            ret, frame = Cam.read()
            frame = alignment(frame)
            ims[:, :, obs] = zoom(cv2.flip(frame, 0), 0.5)
        for i in range(1,int(Sensor.N_mod/2)):
            slm[:, :, 2 * i] = Sensor.modulate_binary((1080, 1920), pxsize=1)
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                frame = alignment(frame)
                ims[:,:,2 * i+obs]= zoom(cv2.flip(frame, 0), 0.5)
            slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                frame = alignment(frame)
                ims[:,:,2 * i + 1 + obs]= zoom(cv2.flip(frame, 0), 0.5)
    elif slm_type=='SLM':
        slm = np.ones((1024, 1280, Sensor.N_mod))
        slm_display = slmpy.SLMdisplay()
        slm_display.updateArray(slm[:, :, 0])
        print(f"Displaying 1 st SLM pattern")
        for obs in range(Sensor.N_os):
            ret, frame = Cam.read()
            frame = alignment(frame)
            ims[:, :, obs] = zoom(cv2.flip(frame, 0), 1)
        for i in range(1,Sensor.N_mod):
            slm[:,:,i]=Sensor.modulate((1024,1280), pxsize=1)
            slm_display.updateArray(slm[:,:,i])
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                frame = alignment(frame)
                ims[:,:,i+obs]= zoom(cv2.flip(frame, 0), 1)
            print(f"Displaying {i+1} th SLM pattern")
        slm_display.close()
        Cam.release()
    if slm_type =='DMD':
        SLM = Sensor.process_SLM(slm, N, delta3, type="amp")
        SLM[np.abs(SLM) > 0.5] = 1 + 1j*0
        SLM[SLM <= 0.5] = 0 + 1j*0
        #fig = plt.figure(1)
        #ax1 = fig.add_subplot(121)
        #ax2 = fig.add_subplot(122)
        #ax1.imshow(np.abs(SLM[:, :, Sensor.N_os]), vmin=0, vmax=1)
        #ax2.imshow(np.abs(u30), vmin=0, vmax=1)
        #plt.show()
    elif slm_type == 'SLM':
        SLM = Sensor.process_SLM(slm, N, delta3, type="phi")
        #fig = plt.figure(1)
        #ax1 = fig.add_subplot(121)
        #ax2 = fig.add_subplot(122)
        #ax1.imshow(np.angle(SLM[:,:,Sensor.N_os]), vmin=-np.pi, vmax = np.pi)
        #ax2.imshow(np.abs(u30), vmin=0, vmax=1)
        #plt.show()
    #ims = Sensor.gen_ims(u30, SLM, z3, delta3, noise)

    print('\nCaptured images are simulated')
    #reconstruction
    #process the captured image : converting to amplitude and padding if needed
    ims=(ims/255.0).astype(np.float)
    y0 = Sensor.process_ims(ims, N)
    plt.imshow(y0[:,:,Sensor.N_os], vmin=0, vmax=1)
    plt.scatter(N/2,N/2, color='r', marker='.')
    plt.show()
    ##Recon initilization
    T_run_0=time.time()
    u3_est, u4_est, idx_converge = Sensor.WISHrun(y0, SLM, delta3, delta4, plot=False)
    T_run=time.time()-T_run_0
    #phase_rms = cp.corrcoef(cp.ravel(cp.angle(cp.asarray(u40))), cp.ravel(cp.angle(u4_est)))[0,1]
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)
    #phase_RMS =(1/N) * np.array(
    #    [np.linalg.norm((np.angle(u40)-np.angle(np.exp(1j*th)*u4_est))*(np.abs(u40) > 0)) for th in
    #     np.linspace(-np.pi, np.pi, 256)])
    #phase_rms = np.min(phase_RMS)
    #phase_rms =(1/N)*np.linalg.norm((np.angle(u30)-np.angle(u3_est))*(np.abs(u30)>0))
    #print(f"\n Phase RMS is : {phase_rms}")
    #total time
    T= time.time()-T0
    print(f"\n Time spent in the GS loop : {T_run} s")
    print(f"\n Total time elapsed : {T} s")
    fig=plt.figure()
    #ax1 = fig.add_subplot(231)
    #ax2 = fig.add_subplot(232)
    #ax3 = fig.add_subplot(233)
    #ax4 = fig.add_subplot(234)
    #ax5 = fig.add_subplot(236)
    ax3 = fig.add_subplot(131)
    ax4 = fig.add_subplot(132)
    ax5 = fig.add_subplot(133)
    #divider1 = make_axes_locatable(ax1)
    #cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    #divider2 = make_axes_locatable(ax2)
    #cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    #im1=ax1.imshow(np.abs(u40), cmap='viridis', vmin=0, vmax=1)
    #ax1.set_title('Amplitude GT')
    #im2=ax2.imshow(np.angle(u40), cmap='twilight_shifted',vmin=-np.pi, vmax=np.pi)
    #ax2.set_title('Phase GT')
    im3=ax3.imshow(abs(u4_est)**2, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('intensity estimation')
    im4=ax4.imshow(np.angle(u4_est), cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase estimation')
    ax5.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    #fig.colorbar(im1, cax=cax1)
    #fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    plt.show()
if __name__=="__main__":
    main()