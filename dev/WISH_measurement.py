# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi on the 28/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cupy as cp
from scipy.ndimage import zoom
from WISH_lkb import WISH_Sensor
from PIL import Image
from SLM import SLM
import cv2
# from hamamatsu.dcam import dcam, Stream, copy_frame
import EasyPySpin
import PySpin
import pycrafter6500
from scipy import ndimage
import sys
from PIL import Image

# plt.switch_backend("QT5Agg")

# WISH routine


def alignment(frame) -> np.ndarray:
    """
    Detects the center of a frame provided the beam is circular
    :param frame: Input frame
    :return: T the translation matrix to correct the image using OpenCV
    """
    # frame_blurred = cv2.blur(frame, (12, 12))
    ret1, thresh = cv2.threshold(frame, 70, 255, 0)
    thresh_blurred = cv2.blur(thresh, (12, 12))
    M = cv2.moments(thresh_blurred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    T = np.float32([[1, 0, int(frame.shape[1]/2) - cX],
                    [0, 1, int(frame.shape[0]/2) - cY]])
    return T

def capture_ims_flir(Sensor: WISH_Sensor, slm_type: str) -> np.ndarray:
    # Setting up the camera for acquisition
    Cam = EasyPySpin.VideoCapture(0)
    # Cam.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
    Cam.auto_software_trigger_execute = True
    # Cam.set(cv2.CAP_PROP_FPS, 82.0)
    # Cam.set(cv2.CAP_PROP_EXPOSURE, 2500)
    Nx = int(Cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ny = int(Cam.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    delta3x = Sensor.wavelength * Sensor.z / (Nx * Sensor.d_CAM)
    delta3y = Sensor.wavelength * Sensor.z / (Ny * Sensor.d_CAM)
    ims = np.zeros((Ny, Nx, Sensor.Nim))
    if slm_type == 'DMD':
        # camera trig for DMD ?
        resX, resY = 1920, 1080
        slm_display = SLM(resX, resY)
        slm = np.zeros((resY, resX, Sensor.N_mod), dtype=np.uint8)
        for i in range(Sensor.N_mod//2):
            slm[:, :, 2 * i] = 2*Sensor.modulate_binary((1080, 1920), pxsize=8)
            slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
        if Sensor.N_os > 1:
            slm_to_display = 255*slm.repeat(Sensor.N_os, axis=2)
        else : 
            slm_to_display = 255*slm

        # dlp = pycrafter6500.dmd()
        # Load images into DMD
        images = [slm_to_display[:, :, i].astype(np.uint8) for i in range(slm_to_display.shape[2])]
        # images = []
        # for i in range(slm_to_display.shape[2]):
        #     im = np.asarray(Image.open(f"intensities/{i%8 + 1}.bmp"))[:, :, 0]
        #     images.append(im)
        del slm_to_display
        for i in range(Sensor.Nim):
            sys.stdout.write(f"\r{i+1}/{Sensor.Nim}")
            slm_display.update(images[i], delay=200)
            ret, frame = Cam.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ims[:, :, i] = frame
        # dlp.stopsequence() # stop the current sequence
        # dlp.changemode(3) # change mode to "pattern on the fly"
        # exp_time = Cam.get(cv2.CAP_PROP_EXPOSURE)*1e-6 #in mus
        # fps = 45.0
        # exposure=[int((1/fps)*1e6)]*len(images)
        # dark_time=[0]*len(images)
        # trigger_in=[False]*len(images)
        # trigger_out=[105]*len(images)
        # dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,0)
        # dlp.startsequence()
        # for i in range(Sensor.Nim):
        #     dlp.startsequence()
        #     if i > 0:
        #         time.sleep(exposure[0]*1e-6) # starts from the last displayed pattern so wait for one exposure time
        #     ret, frame = Cam.read()
        #     ims[:, :, i] = frame
        #     sys.stdout.write(f"\rCaptured frame {i+1}/{Sensor.Nim}")
        #     dlp.pausesequence()
        #     # wait for next possible frame (83 fps=16ms)
        #     time.sleep(1/fps)
        # dlp.stopsequence()
        slm_display.close()
    elif slm_type == 'SLM':
        resX, resY = 1920, 1080
        corr = np.zeros((resY, resX))
        slm = np.zeros((resY, resX, Sensor.N_mod), dtype=np.uint8)
        for i in range(slm.shape[2]):
            slm[:, :, i] = Sensor.modulate((resY, resX), pxsize=8)
        slm_display = SLM(resX, resY)
        for i in range(Sensor.N_mod):
            slm_pic = (250*((1/255)*(256*slm[:, :, i]-corr)%256)).astype('uint8')
            slm_display.update(slm_pic)
            for obs in range(Sensor.N_os):
                ret, frame = Cam.read()
                ims[:, :, Sensor.N_os*i + obs] = frame
        slm_display.close()
    Cam.release()
    return ims, slm, Nx, Ny


# def capture_ims_hamamatsu(Sensor: WISH_Sensor, slm_type: str) -> np.ndarray:
#     # Setting up the camera for acquisition
#     with dcam:
#         camera = dcam[0]
#         with camera:
#             Nx, Ny = camera['image_width'].value, camera['image_height'].value
#     delta3x = Sensor.wavelength * Sensor.z / (Nx * Sensor.d_CAM)
#     delta3y = Sensor.wavelength * Sensor.z / (Ny * Sensor.d_CAM)
#     ims = np.zeros((Ny, Nx, Sensor.Nim))
#     if slm_type == 'DMD':
#         resX, resY = 1920, 1080
#         # if N_os > 1, slm needs to be of size (Ny, Nx, Nim)
#         slm = np.zeros((resY, resX, Sensor.N_mod), dtype=np.uint8)
#         for i in range(0, Sensor.N_mod//2):
#             slm[:, :, 2 * i] = 2*Sensor.modulate_binary((1080, 1920), pxsize=1)
#             slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
#         slm_to_display = slm.repeat(Sensor.N_os, axis=2)
#         dlp = pycrafter6500.dmd()
#         # Load images into DMD
#         images = [slm_to_display[:, :, i].astype(np.uint8) for i in range(slm.shape[2])]
#         del slm_to_display
#         dlp.stopsequence() # stop the current sequence
#         dlp.changemode(3) # change mode to "pattern on the fly"
#         exposure=[50000]
#         dark_time=[0]
#         trigger_in=[False]
#         trigger_out=[105]
#         print(f"Loading images on the DMD ...")
#         dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,0)
#         with dcam:
#             camera = dcam[0]
#             with camera:
#                 print(f"Acquiring images ...")
#                 with Stream(camera, Sensor.Nim) as stream:
#                     dlp.startsequence()
#                     camera.start() #will this be synchronized enough ? Maybe need to add a first blank pattern if the cam skips the first trig point
#                     for i, frame_buffer in enumerate(stream):
#                         sys.stdout.write(f"\rImage {i+1}/{Sensor.Nim} acquired")
#                         ims[:, :, i] = copy_frame(frame_buffer)
#     elif slm_type == "SLM":
#         resX, resY = 1920, 1080
#         slm = np.zeros((resY, resX, Sensor.N_mod), dtype=np.uint8)
#         for i in range(slm.shape[2]):
#             slm[:, :, i] = Sensor.modulate((resY, resX), pxsize=8)
#         slm_display = SLMscreen(resX, resY)
#         with dcam:
#             camera = dcam[0]
#             with camera:
#                 print(f"Acquiring images ...")
#                 for i in range(Sensor.N_mod):
#                     slm_display.update(slm[:, :, i])
#                     time.sleep(0.15)
#                     with Stream(camera, Sensor.N_os) as stream:
#                         for k, frame_buffer in enumerate(stream):
#                             sys.stdout.write(f"\rImage {i+1}/{Sensor.Nim} acquired")
#                             ims[:, :, k] = copy_frame(frame_buffer)
#     return ims, slm, Nx, Ny


def main():
    # start timer
    T0 = time.time()
    # instantiate WISH sensor reading the conf file
    Sensor = WISH_Sensor("wish_3.conf")
    # load SLM flatness correction
    # corr = Image.open("/home/tangui/Documents/SLM/deformation_correction_pattern/CAL_LSH0802200_780nm.bmp")
    # corr = Image.open("/home/tangui/Documents/phase_retrieval/dev/phases/U14-2048-201133-06-04-07_808nm.png")
    # corr = np.zeros((1080, 1920))
    # wvl = Sensor.wavelength
    # z3 = Sensor.z
    # delta4x = Sensor.d_CAM
    # delta4y = Sensor.d_CAM
    slm_type = 'DMD'
    # if slm_type == 'DMD':
    # ims, slm, Nx, Ny = capture_ims_hamamatsu(Sensor, slm_type)
    ims, slm, Nx, Ny = capture_ims_flir(Sensor, slm_type)

    # elif slm_type == 'SLM':
    #     # Setting up the camera for acquisition
    #     Cam = EasyPySpin.VideoCapture(0)
    #     Nx = int(Cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     Ny = int(Cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     # Cam.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
    #     # Cam.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    #     # Cam.set(cv2.CAP_PROP_FPS, 88.0)
    #     delta3x = wvl * z3 / (Nx * delta4x)
    #     delta3y = wvl * z3 / (Ny * delta4y)
    #     ims = np.zeros((Ny, Nx, Sensor.Nim))
    #     resX, resY = 1920, 1080
    #     angle = 0  # to correct for misalignment btwn camera and SLM
    #     slm_display = SLMscreen(resX, resY)
    #     slm = np.ones((resY, resX, Sensor.N_mod))
    #     slm_display.update(slm[:, :, 0].astype('uint8'))
    #     ret, frame = Cam.read()
    #     frame = cv2.flip(frame, 0)
    #     mask = frame > Sensor.threshold*np.max(frame)
    #     for i in range(Sensor.N_mod):
    #         slm[:, :, i] = Sensor.modulate((resY, resX), pxsize=8)
    #         print(f"Displaying {i} th SLM pattern")
    #         slm_pic = (250*((1/255)*(256*slm[:, :, i]-corr)%256)).astype('uint8')
    #         slm_display.update(slm_pic)
    #         time.sleep(0.15)
    #         t0 = time.time()
    #         for obs in range(Sensor.N_os):
    #             # time.sleep(0.05)
    #             ret, frame = Cam.read()
    #             # frame = cv2.warpAffine(frame, T, frame.shape)
    #             frame = cv2.flip(frame, 0)
    #             # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #             # frame = ndimage.rotate(frame, -angle, reshape=False)
    #             # ims[:,:,Sensor.N_os*i+obs]= zoom(frame, zoom_factor)
    #             print(f"Recording image nbr : {Sensor.N_os*i+obs}")
    #             ims[:, :, Sensor.N_os*i+obs] = frame*mask
    #         t = time.time()-t0
    #         print(f"Took {t} s to grab the frames")
    #     slm_display.close()
    #     Cam.release()
    if slm_type == 'DMD':
        SLM = Sensor.process_SLM(slm, Nx, Ny, Sensor.d_CAM, Sensor.d_CAM, type="amp")
        # DMD angle corrections
        x = (SLM.shape[1]//2 - np.linspace(0, SLM.shape[1]-1, SLM.shape[1]))
        y = (SLM.shape[0]//2 - np.linspace(0, SLM.shape[0]-1, SLM.shape[0]))
        X, Y = np.meshgrid(x, y)
        alpha = 5.0 * (np.pi/180) # angle of incidence
        theta = 12.0 * (np.pi/180) # angle of DMD mirrors 
        theta_1D = np.arctan(np.tan(theta)/np.sqrt(2))
        beta = -alpha +2*theta # reflection angle
        alpha_1D = np.arctan(np.tan(alpha)/np.sqrt(2))
        beta_1D = 2*theta_1D-alpha_1D#np.arctan(np.tan(beta)/np.sqrt(2))
        dmd_blaz_corr = np.exp((X-Y)*complex(0,1)*2*np.pi/Sensor.wavelength*Sensor.d_SLM*(np.sin(alpha_1D)+np.sin(beta_1D)))
        SLM = np.array([SLM[:, :, i]*dmd_blaz_corr for i in range(SLM.shape[2])]).transpose((1, 2, 0))
        # plt.imshow(X1)
        # plt.show()
    elif slm_type == 'SLM':
        SLM = Sensor.process_SLM(slm, Nx, Ny, Sensor.d_CAM, Sensor.d_CAM, type="phi")
    # plt.imshow(np.abs(SLM[:, :, 0]))
    # plt.show()
    print('\nModulated images are captured')
    # reconstruction
    # process the captured image : converting to amplitude and padding if needed
    ims = (ims/(2**16)).astype(np.float32)
    print('\nDisplaying captured images')
    y0 = Sensor.process_ims(ims, Nx, Ny)
    # for k in range(y0.shape[2]):
    ncol = 6
    fig, ax = plt.subplots(4, ncol)
    for i in range(Sensor.Nim):
        ax[i//ncol, i%ncol].imshow(y0[:, :, i], vmin=0, vmax=1)
        ax[i//ncol, i%ncol].set_title(f"{i+1}")
    plt.show()
    # Recon initilization
    T_run_0 = time.time()
    delta3x = Sensor.wavelength * Sensor.z / (Nx * Sensor.d_CAM)
    delta3y = Sensor.wavelength * Sensor.z / (Ny * Sensor.d_CAM)
    u3_est, u4_est, idx_converge = Sensor.WISHrun_vec(y0, SLM, delta3x,
                                                      delta3y, Sensor.d_CAM,
                                                      Sensor.d_CAM)
    # np.save("u4_est.npy", u4_est)
    T_run = time.time()-T_run_0
    # Backpropagation distance for the 2nd plot
    # z2 = 130e-3
    # u2_est = Sensor.frt_gpu(u4_est, Sensor.d_CAM, Sensor.d_CAM,
    #                         Sensor.wavelength, Sensor.n, -z2)
    # Z2 = np.linspace(170e-3, 195e-3, 25)
    # for z2 in Z2:
    #     u2_est = Sensor.frt_gpu(u4_est, Sensor.d_CAM, Sensor.d_CAM,
    #                             Sensor.wavelength, Sensor.n, -z2)
    #     u2_est = cp.asnumpy(u2_est)
    #     plt.imshow(np.abs(u2_est))
    #     plt.title(f"{1e3*z2} mm")
    #     plt.show()
    # u2_est = cp.asnumpy(u2_est)
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)
    # total time
    T = time.time()-T0
    print(f"\n Time spent in the GS loop : {T_run} s")
    print(f"\n Total time elapsed : {T} s")
    fig = plt.figure()
    ax3 = fig.add_subplot(141)
    ax4 = fig.add_subplot(142)
    ax6 = fig.add_subplot(143)
    ax5 = fig.add_subplot(144)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    divider6 = make_axes_locatable(ax6)
    cax6 = divider6.append_axes('right', size='5%', pad=0.05)
    im3 = ax3.imshow(abs(u4_est), cmap='viridis') #, vmin=0, vmax=1)
    ax3.set_title('intensity estimation (camera plane)')
    im4 = ax4.imshow(np.angle(u4_est), cmap='twilight_shifted', vmin=-np.pi,
                     vmax=np.pi)
    ax4.set_title('Phase estimation')
    im6 = ax6.imshow(np.abs(u3_est), cmap='viridis') #, vmin=0, vmax=1)
    ax6.set_title('Back-propagated field (SLM plane)')
    ax5.plot(5*np.arange(0, len(idx_converge), 1), idx_converge)
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    ax5.set_yscale('log')
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    fig.colorbar(im6, cax=cax6)
    plt.show()
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121)
    # ax1 = fig.add_subplot(122)
    # divider0 = make_axes_locatable(ax0)
    # divider1 = make_axes_locatable(ax1)
    # cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    # im0 = ax0.imshow(abs(u2_est) ** 2, cmap='viridis')  # , vmin=0, vmax=1)
    # im1 = ax1.imshow(np.angle(u2_est), cmap='twilight_shifted', vmin=-np.pi,
    #                  vmax=np.pi)
    # ax0.set_title("Back-propagated intensity")
    # ax1.set_title("Back-propagated phase")
    # fig.colorbar(im0, cax0)
    # fig.colorbar(im1, cax1)
    # plt.show()


if __name__ == "__main__":
    main()
