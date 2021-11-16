import pycrafter6500
import numpy as np
import PIL.Image
from WISH_lkb import WISH_Sensor, WISH_Sensor_cpu
import matplotlib.pyplot as plt
import time
import sys

def circle(R, width, value):
    """
    Draws a circle
    :param R: Radius of the circle
    :param width: Width of the circle in px
    :param value : Value inside the circle
    :return: The circle as a 2d array
    """
    x = 1920//2 - np.linspace(0, 1919, 1920)
    y = 1080//2 - np.linspace(0, 1079, 1080)
    X, Y = np.meshgrid(x, y)
    out = np.zeros(X.shape, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii > (R-width/2)
    cond &= Radii < (R+width/2)
    out[cond] = value
    return out

def cross(x0, y0, width):
    x = 1920//2 - np.linspace(0, 1919, 1920)
    y = 1080//2 - np.linspace(0, 1079, 1080)
    X, Y = np.meshgrid(x, y)
    out = np.zeros(X.shape, dtype='uint8')
    out[x0-width//2:x0+width//2, :] = 255
    out[:, y0-width//2:y0+width//2] = 255
    return out

def checkerboard(gridsize:int = 20):
    """Defines a square checkerboard pattern for camera alignment

    Args:
        gridsize (int, optional): Size of the board squares. Defaults to 20.
    """
    x = np.zeros((1080, 1920), dtype=bool)
    x[1::2, ::2] = True
    x[::2, 1::2] = True
    return x

def Checkerboard(sq:int = 20):
    """Defines a square checkerboard pattern for camera alignment

    Args:
        sq (int, optional): Size of the board squares. Defaults to 20.
    """
    h, w = 1080, 1920
    pix = np.zeros((h, w, 3), dtype=np.uint8)
    # Make a checkerboard
    row = [[(1000,1000,1000),(0,0,0)][(i//sq)%2] for i in range(w)]
    pix[[i for i in range(h) if (i//sq)%2 == 0]] = row
    row = [[(0,0,0),(1000,1000,1000)][(i//sq)%2] for i in range(w)]
    pix[[i for i in range(h) if (i//sq)%2 == 1]] = row
    return pix

A=checkerboard()
plt.imshow(A)
plt.show()

B=Checkerboard()
plt.imshow(B)
plt.show()

Sensor = WISH_Sensor("wish_3.conf")
slm = np.zeros((1080, 1920, Sensor.N_mod), dtype=np.uint8)
for i in range(0, Sensor.N_mod):
    slm[:, :, i] = 2*Sensor.modulate_binary((1080, 1920), pxsize=10)
# # slm[:, :, 1] = 2*Sensor.modulate_binary((1080, 1920), pxsize=1)
# for i in range(0, Sensor.N_mod//2):
#     slm[:, :, 2 * i] = 2*Sensor.modulate_binary((1080, 1920), pxsize=1)
#     slm[:, :, 2 * i + 1] = np.ones((1080, 1920)) - slm[:, :, 2 * i]
images = []
# im = np.asarray(PIL.Image.open("intensities/harambe.tif"))[:, :, 0]//129
# im = np.asarray(PIL.Image.open("intensities/alignment_dmd.bmp"))[:, :, 0]//129
# im0 = np.asarray(PIL.Image.open("intensities/resChart.bmp"))[:, :, 0]//129
# paddingy = int((1080-512)/2)
# paddingx = int((1920-512)/2)
# im = np.pad(im0, ((paddingy, paddingy),(paddingx, paddingx)))
# images.append(im)
# print(np.asarray(PIL.Image.open("intensities/harambe.tif"))[:, :, 0]//129)
# images = [slm[:, :, i].astype(np.uint8) for i in range(slm.shape[2])]
# images = [np.ones((slm.shape[0], slm.shape[1]), dtype=np.uint8)] # off
# images = [np.zeros((slm.shape[0], slm.shape[1]), dtype=np.uint8)] # on
# images = [circle(150, 10, 255)]
images = [cross(1080//2, 1920//2, 20)]
dlp = pycrafter6500.dmd()

dlp.stopsequence()

dlp.changemode(3)

exposure=[105]*len(images)
dark_time=[0]*len(images)
trigger_in=[False]*len(images)
trigger_out=[105]*len(images)

dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,0)
ts = np.empty(len(images), dtype=np.float64)
dlp.startsequence()
# for i in range(len(images)):
#     tic = time.perf_counter()
#     dlp.startsequence()
#     time.sleep(exposure[0]*1e-6)
#     toc = time.perf_counter()
#     ts[i] = toc-tic
#     dlp.pausesequence()
# print("")
# plt.plot(ts)
# plt.show()