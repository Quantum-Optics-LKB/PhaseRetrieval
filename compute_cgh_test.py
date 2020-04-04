# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:26:08 2016

@author: Tangui ALADJIDI
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from timeit import default_timer as timer
import random
import os


calib=plt.imread('LSH0500427_850nm_calibration.bmp') #calibration phase map of the SLM
calib=calib*0.031*np.ones(calib.shape)-0.29*np.ones(calib.shape)
R=np.ones(calib.shape)
lamb=795*10**(-9)
k=2*np.pi/lamb
for i in range(len(R[:,0])):
    for j in range(len(R[0,:])):
        R[i,j]=np.sqrt(((i-300)*20*10**(-6))**2+((j-396)*20*10**(-6))**2)
Zet=(k*R)/0.75
#Defining the phase term associated with the Helmholtz propagator
def theta_helm(z):
    out=np.exp(1j*k*z*np.sqrt(1-(lamb*Zet)**2))
    return out
def lens(f,R):
    L=np.exp(1j*(k*R**2/2*f))
    return L
    
def compute_cgh(alg,i_tgt,p_output,**kwargs):
    """

    alg : Algorithm choice either MRAF or OMRAF

    i_tgt : target intensity (png image)

    p_output : name of the file in which to save the phase profile (dummy variable, this test code does not save anything)

    **iterations : number of iterations of the GS loop (default is 5

    **mixing : mixing parameter between two iterations. Next iteration is m*iteration(n)+(1-m)*iteration(n-1)
    
    **offset : introduces an offset to shift the lowest intensity to a non-zero value. Improves convergence.

    **propagator : propagator for the simulation of light propagation either angular formalism ("Helmholtz") or simple Fourier transform ("Fourier"). Default is "Fourier".

    **phi_0 : initial phase guess (PNG image file). Can also be lens(x) where x is the focal length. In this case the iniial phase will be the parabolic profile of a lens of focal length x. This usually improves convergence a bit and avoids vortices.

    **pad : boolean. Does zero padding. Do not remember exactly how I implement it but must be useful somehow for sampling reasons.

    """
    start=timer()
    for args in kwargs:
                print(args,'',kwargs[args])
    R=np.ones(calib.shape)
    for i in range(len(R[:,0])):
        for j in range(len(R[0,:])):
            R[i,j]=np.sqrt(((i-300)*20*10**(-6))**2+((j-396)*20*10**(-6))**2)
    if alg in ['MRAF','OMRAF']:
    #Target intensity
        i_tgt=plt.imread(str(i_tgt))
        if i_tgt.ndim==3:
            print('Multichannel target intensity found : taking the first channel')
            i_tgt=i_tgt[:,:,0]
        elif i_tgt.ndim>3:
            print('Non-compatible target intensity image specified')
    #Initial intensity   
        if ('i_0' in kwargs):
            if kwargs['i_0']!='':
                i_0=plt.imread(kwargs['i_0'])
            else :
                i_0=np.exp(-(R/(np.sqrt(2)*0.001))**2)
        else :
            i_0=np.exp(-(R/(np.sqrt(2)*0.001))**2)
        if i_0.ndim==3:
            print('Multichannel initial intensity foud : taking the first channel')
            i_0=i_0[:,:,0]
        elif i_0.ndim>3:
            print('Non-compatible initial intensity image specified')
    #Iterations number
        if ('iterations' in kwargs):
            if kwargs['iterations']!='':        
                iterations=int(kwargs['iterations'])
            else :
                iterations=5
        else :
            iterations=5
    #Mixing parameter       
        if ('mixing' in kwargs):
            if kwargs['mixing']!='':
                m=float(kwargs['mixing'])
            else :
                m=0.9
        else :
            m=0.9
    #Canevas region        
        if ('mask' in kwargs):
            if kwargs['mask']!='':
                mask_sr = plt.imread(str(kwargs['mask'])) #Defining the mask's noise and signal region
                mask_sr = np.array(mask_sr)
                mask_sr[mask_sr==255]=1
                mask_nr = plt.imread(str(kwargs['mask']))
                mask_nr = np.array(mask_nr)
                mask_nr[mask_nr==0]=1
                mask_nr[mask_nr==255]=0
            else :
                print('You need to specify a mask !')
        else :
            print('You need to specify a mask !')
#Optional padding of the SLM plane to get up to 1584x1200 size        
        if ('pad' in kwargs):
            if float(kwargs['pad'])==1:
#Now we need to convert all of the arrays to 1584x1200 according to Nyquist's theorem
                tmp=np.zeros((1200,1584))
                tmp[300:900,396:1188]=mask_sr
                mask_sr=tmp
                tmp=np.ones((1200,1584))
                tmp[300:900,396:1188]=mask_nr
                mask_nr=tmp
                tmp=np.zeros((1200,1584))
                tmp[300:900,396:1188]=i_tgt
                i_tgt=tmp
                R=np.zeros((1200,1584))
                for i in range(len(R[:,0])):
                    for j in range(len(R[0,:])):
                        R[i,j]=np.sqrt(((i-600)*20*10**(-6))**2+((j-792)*20*10**(-6))**2)
                i_0=np.exp(-(R/(np.sqrt(2)*0.001))**2)
                Zet=(k*R)/0.75
                
        
      #Initial phase pattern  
        if ('phi_0' in kwargs):
            if kwargs['phi_0']!='':
            #Checks if the phi_0 (initial phase) argument takes the form of 'lens(x)', if yes the 
            #the initial phase will be the Fresnel lens of focal length x, else the 
            #argument is a file name and the initial phase will be the specified file
                if "lens(" in str(kwargs['phi_0']):
                    phi=lens(float(kwargs['phi_0'][5:len(kwargs['phi_0'])-1]),R)
                else :
                    phi=plt.imread(kwargs['phi_0'])
            else :
                phi=np.angle(lens(2.5,R))
        else :
            phi=np.angle(lens(0.75,R))
        if phi.ndim==3:
            print('Multichannel initial phase found : taking the first channel')
            phi=phi[:,:,0]
        elif phi.ndim>3:
            print('Non-compatible initial phase image specified (too many dimensions)')
        elif phi.shape!=i_tgt.shape:
            print('Non-compatible initial phase image specified (wrong size)')

	#Core of the calculation
        i=0
        if alg=='MRAF':
            print('MRAF')
            if kwargs["propagator"]=="Fourier":
                print("Fourier")
                while i<iterations:
                    E_n_out=np.fft.fft2(i_0*np.exp(1j*phi),norm="ortho")
                    E_n_out_=np.fft.fftshift(E_n_out)
                    G_n=(m*i_tgt*mask_sr)*np.exp(1j*np.angle(E_n_out_))+(1-m)*E_n_out_*mask_nr
                    phi=np.angle(np.fft.ifft2(G_n))
                    i_f=abs(E_n_out)**2
                    i=i+1
                    print((float(i)/iterations)*100," % done ...")
            else :
                print("Helmholtz")
                theta=theta_helm(0.695)
                theta1=theta_helm(0.75)
                lens0=lens(0.75)
                while i<iterations:
                    E_n_out=np.fft.fft2(i_0*np.exp(1j*phi),norm="ortho")
                    E_n_out_=np.fft.fftshift(E_n_out)
                    E_n_out_=E_n_out_*theta
                    E_n_out_=fftconvolve(E_n_out_,np.fft.fftshift(np.fft.fft2(lens0)),mode="same")
                    E_n_out_=E_n_out_*theta1
                    G_n=(m*i_tgt*mask_sr)*np.exp(1j*np.angle(E_n_out_))+(1-m)*E_n_out_*mask_nr
                    phi=np.angle(np.fft.ifft2(G_n))
                    i_f=abs(E_n_out)**2
                    i=i+1
                    print((float(i)/iterations)*100," % done ...")
                
            phase_img = (phi+calib)%(2*np.pi)
            i_f=((i_f/np.max(i_f).astype(float))*255).astype(np.uint8)
            phase_img=((phase_img/0.031)+(0.29/0.031)*np.ones(i_tgt.shape)).astype(np.uint8)
            phase_img=Image.fromarray(phase_img)
            print('Saving phase file ..')
            phase_img.save(p_output)
            if ('a_out' in kwargs):
                intensity_img=i_f
                intensity_img=Image.fromarray(intensity_img)
                intensity_img.save(kwargs['a_out'])
        elif alg=='OMRAF':
            print('OMRAF')
            if ('offset' in kwargs):
                if kwargs['offset']!='':
                    offset=float(kwargs['offset'])
                else :
                    offset=5.
            else :
                offset=5.
            if kwargs["propagator"]=="Fourier":
                print("Fourier")
                while i<iterations:
                    if i==0:
                        E_n_out=np.fft.fft2(i_0*np.exp(1j*phi))
                    else :
                        E_n_out=np.fft.fft2(np.exp(1j*phi))
                    E_n_out_=np.fft.fftshift(E_n_out)
                    G_n=(m*(i_tgt+(np.ones(i_tgt.shape)*offset).astype(np.uint8)*mask_sr))*np.exp(1j*np.angle(E_n_out_))+(1-m)*E_n_out_*mask_nr
                    phi=np.angle(np.fft.ifft2(G_n))
                    i_f=abs(E_n_out)**2
                    i=i+1
                    print((float(i)/iterations)*100," % done ...")
            else :
                print("Helmholtz")
                theta=theta_helm(0.695)
                theta1=theta_helm(0.75)
                lens0=lens(0.75)
                while i<iterations:
                    E_n_out=np.fft.fft2(i_0*np.exp(1j*phi),norm="ortho")
                    E_n_out_=np.fft.fftshift(E_n_out)
                    E_n_out_=E_n_out_*theta
                    E_n_out_=fftconvolve(E_n_out_,np.fft.fftshift(np.fft.fft2(lens0)),mode="same")
                    E_n_out_=E_n_out_*theta1
                    E_n_out_=np.fft.ifft2(E_n_out_)
                    G_n=(m*(i_tgt+(np.ones(i_tgt.shape)*offset).astype(np.uint8)*mask_sr))*np.exp(1j*np.angle(E_n_out_))+(1-m)*E_n_out_*mask_nr
                    phi=np.angle(np.fft.ifft2(G_n))
                    i_f=abs(E_n_out)**2
                    i=i+1
                    print((float(i)/iterations)*100," % done ...")
   #Converting all the angles to image format via the look up table                                
            phase_img = (phi[300:900,396:1188]+calib)%(2*np.pi)
            i_f=((i_f/np.max(i_f).astype(float))*255).astype(np.uint8)
            phase_img=((phase_img/0.031)+(0.29/0.031)*np.ones(phase_img.shape)).astype(np.uint8)
            
        print("Time elapsed : ",timer()-start," seconds")
        return phase_img,i_f
    else :
        print('This function only supports MRAF and OMRAF algorithms')
    

phi,i_f=compute_cgh('OMRAF','TransistorIoSLM.bmp','t.bmp',iterations=20,mixing=0.99,mask='TransistorIoSLM25_mask.bmp',offset=5,propagator='Fourier',pad=1)
plt.subplot(121)
plt.title("Generated phase map")
plt.imshow(phi,cmap=plt.cm.gray)
plt.subplot(122)
plt.title("Estimated intensity map")
plt.imshow(i_f, cmap=plt.cm.gray)
plt.show()
