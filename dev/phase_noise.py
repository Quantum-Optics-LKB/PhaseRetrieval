# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_volt


P = 0.00000004 # W beam power
T = 1e-6 #s pulse duration
wx = 0.4e-3 #m pulse waist
wy = 0.4e-3 #m pulse waist
N = 1024 #image resolution
dx = 1.85e-6 #m camera pixel pitch
dy = 1.85e-6 #m camera pixel pitch

def poisson(lam, k):
    P_k = np.exp(-lam) * ((lam ** k) / np.math.factorial(k))
    return P_k


def P_phi(phi, lam):
    N = np.arange(0, 60, 1)  # indices for the sum
    P_n = np.array([poisson(lam, n) for n in N])
    e_n = np.exp(1j*phi*N)
    P = (1 / (2 * np.pi)) * np.abs(np.sum(P_n * e_n)) ** 2
    return P

def input_state(P: float, wx: float, wy: float, dt: float ):
    """
    Aa function to generate arbitrary coherent pulsed input states
    :param P: Initial power in W
    :param wx: Beam waist along x in m
    :param wy: Beam waist along y in m
    :param dt: Pulse length
    :return: Psi  np.ndarray initial state
    """
    I0 = 2*P/(np.pi*wx*wy) #initial intensity
    x = dx*np.arange(-N/2,N/2,1)
    y = dy*np.arange(-N/2,N/2,1)
    X, Y = np.meshgrid(x,y)
    I = I0 * np.exp(-2*(X**2 + Y**2)/(wx*wy))
    N_mean = I*dx*dy*dt/electron_volt #for now I just took 1 ev, but could be anything
    dI = np.random.poisson(N_mean)*(electron_volt/(dx*dy*dt)) - I
    phi = np.linspace(-np.pi, np.pi, 32)
    Phi = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            Phi[i,j]=np.random.choice(phi, [P_phi(p, N_mean[i,j]) for p in phi])
    plt.imshow(Phi, vmin=-np.pi, vmax=np.pi)
    plt.show()
    Psi = np.sqrt(I+ dI)*np.exp(1j*dphi)
    return Psi
Psi=input_state(P, wx, wy, T)
plt.imshow(Psi)
plt.show()