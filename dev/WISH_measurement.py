# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 28/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cupy as cp
from scipy.ndimage import zoom
from WISH_lkb import WISH_Sensor
from SLM import SLMscreen
import cv2
import EasyPySpin

plt.switch_backend("QT5Agg")

#WISH routine
def alignment(frame):
    frame_blurred = cv2.blur(frame, (12, 12))
    ret1, thresh = cv2.threshold(frame, 70, 255, 0)
    thresh_blurred = cv2.blur(thresh, (12, 12))
    M = cv2.moments(thresh_blurred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    T = np.float32([[1, 0, int(frame.shape[1]/2) - cX], [0, 1, int(frame.shape[0]/2) - cY]])
    return T
def main():
    #start timer
    T0 = time.time()
    #instantiate WISH
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    slm_type = 'SLM'
    #Setting up the camera for acquisition
    Cam = EasyPySpin.VideoCapture(0) #by default camera 0 is the laptop webcam
    zoom_factor=1
    N = int(Cam.get(cv2.CAP_PROP_FRAME_WIDTH)*zoom_factor)
    delta3 = wvl * z3 / (N * delta4)
    ims = np.zeros((N,N,Sensor.Nim))
    if slm_type=='DMD':
        slm = np.ones((1080, 1920, Sensor.N_mod))
        slm_display = SLMscreen(1920,1080)
        slm_display.update(slm[:, :, 0])
        print(f"Displaying 1 st SLM pattern")
        for obs in range(Sensor.N_os):
            ret, frame = Cam.read()
            T = alignment(frame)
            frame = cv2.warpAffine(frame, T, frame.shape)
            ims[:, :, obs] = zoom(cv2.flip(frame, 0), zoom_factor)
        for i in range(1,int(Sensor.N_mod/2)):
            slm[:, :, 2 * i] = Sensor.modulate_binary((1080, 1920), pxsize=1)
            slm_display.update(slm[:, :, 2 * i])
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                frame = alignment(frame)
                ims[:,:,2 * Sensor.N_os*i+obs]= zoom(cv2.flip(frame, 0), zoom_factor)
            slm[:, :, 2 *i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
            slm_display.update(slm[:, :, 2 * i+1])
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                frame = alignment(frame)
                ims[:,:,2 * i + 1 + obs]= zoom(cv2.flip(frame, 0), 0.5)
    elif slm_type=='SLM':
        resX, resY = 1280, 1024
        slm_display = SLMscreen(resX,resY)
        slm = np.ones((resY, resX, Sensor.N_mod))
        slm_display.update(slm[:, :, 0].astype('uint8'))
        print(f"Displaying 1 st SLM pattern")
        for obs in range(Sensor.N_os):
            ret, frame = Cam.read()
            if obs==0:
                T = alignment(frame)
            #frame = cv2.warpAffine(frame, T, frame.shape)
            ims[:, :, obs] = zoom(cv2.flip(frame, 0), zoom_factor)
        for i in range(1,Sensor.N_mod):
            slm[:,:,i]=Sensor.modulate((resY,resX), pxsize=1)
            slm_display.update((256*slm[:,:,i]).astype('uint8'))
            print(f"Displaying {i + 1} th SLM pattern")
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                #frame = cv2.warpAffine(frame, T, frame.shape)
                ims[:,:,Sensor.N_os*i+obs]= zoom(cv2.flip(frame, 0), zoom_factor)
        slm_display.close()
        Cam.release()
    if slm_type =='DMD':
        SLM = Sensor.process_SLM(slm, N, delta3, type="amp")
        SLM[np.abs(SLM) > 0.5] = 1 + 1j*0
        SLM[SLM <= 0.5] = 0 + 1j*0
    elif slm_type == 'SLM':
        SLM = Sensor.process_SLM(slm, N, delta3, type="phi")

    print('\nCaptured images are simulated')
    #reconstruction
    #process the captured image : converting to amplitude and padding if needed
    ims=(ims/256).astype(np.float)
    print('\nDisplaying captured images')
    y0 = Sensor.process_ims(ims, N)
    for k in range(y0.shape[2]):
        plt.imshow(y0[:,:,k], vmin=0, vmax=1)
        plt.scatter(N/2,N/2, color='r', marker='.')
        plt.show()
    ##Recon initilization
    T_run_0=time.time()
    u3_est, u4_est, idx_converge = Sensor.WISHrun(y0, SLM, delta3, delta4, plot=False)
    T_run=time.time()-T_run_0
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)
    #total time
    T= time.time()-T0
    print(f"\n Time spent in the GS loop : {T_run} s")
    print(f"\n Total time elapsed : {T} s")
    fig=plt.figure()
    ax3 = fig.add_subplot(131)
    ax4 = fig.add_subplot(132)
    ax5 = fig.add_subplot(133)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    im3=ax3.imshow(abs(u4_est)**2, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('intensity estimation')
    im4=ax4.imshow(np.angle(u4_est), cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase estimation')
    ax5.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    ax5.set_yscale('log')
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    plt.show()
if __name__=="__main__":
    main()