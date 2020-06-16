# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 16/06/2020
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import cupy as cp
from WISH_lkb import WISH_Sensor

plt.switch_backend("QT5Agg")
def main():
##Recon initilization
    SLM=np.load("measurements/SLM.npy")
    y0 = np.load("measurements/y0.npy")
    Sensor = WISH_Sensor("wish_3.conf")
    wvl = Sensor.wavelength
    z3 = Sensor.z
    delta4 = Sensor.d_CAM
    N = y0.shape[0]
    delta3 = wvl * z3 / (N * delta4)
    T0=time.time()
    u3_est, u4_est, idx_converge = Sensor.WISHrun(y0, SLM, delta3, delta4, plot=False)
    u3_est = cp.asnumpy(u3_est)
    u4_est = cp.asnumpy(u4_est)
    #total time
    T= time.time()-T0
    print(f"\n Time elapsed : {T} s")
    fig=plt.figure()
    ax3 = fig.add_subplot(131)
    ax4 = fig.add_subplot(132)
    ax5 = fig.add_subplot(133)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    im3=ax3.imshow(abs(u4_est)**2, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('intensity estimation (SLM plane)')
    im4=ax4.imshow(np.angle(u4_est), cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase estimation')
    ax5.plot(np.arange(0, len(idx_converge),1), idx_converge)
    ax5.set_title("Convergence curve")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("RMS error of the estimated field")
    ax5.set_yscale('log')
    fig.colorbar(im3, cax=cax3)
    fig.colorbar(im4, cax=cax4)
    fig.set_size_inches(16,8)
    plt.tight_layout()
    plt.savefig("final.png", dpi=300)
if __name__=="__main__":
    main()