# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 19/03/2021
"""

import cv2
import numpy as np

class SLM:
    def __init__(self, resX:int, resY:int, name: str="SLM"):
        """Initializes the SLM screen

        Args:
            resX (int): Resolution along x axis
            resY (int): Resolution along y axis
            name (str, optional): Name of the SLM window. Defaults to "SLM".
        """
        self.name = name
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(name, resX, 0)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN)
        self.update(np.ones((resY, resX), dtype=np.uint8))
    def update(self, A: np.ndarray, delay: int=1):
        """Updates the pattern on the SLM

        Args:
            A (np.ndarray): Array
            delay (int, optional): Delay in ms. Defaults to 1.
        """
        assert A.dtype == np.uint8, "Only 8 bits patterns are supported by the SLM !"
        cv2.imshow(self.name, A)
        cv2.waitKey(1+delay)
    def close(self):
        cv2.destroyWindow(self.name)

def main():
    import sys
    import time
    resX, resY = 1920, 1080
    slm = SLM(resX, resY)
    T = np.zeros(20)
    for i in range(20):
        sys.stdout.flush()
        one = np.ones((resY, resX), dtype=np.uint8)
        slm_pic = (i % 2)*one[:, 0:resX//2] + \
                ((i+1) % 2)*255*one[:, resX//2:]
        t0 = time.time()
        slm.update(slm_pic, delay=250)
        t = time.time()-t0
        T[i] = t
        sys.stdout.write(f"\r{i+1} : time displayed = {t} s")
    slm.close()

    print(f"\nAverage display time = {np.mean(T)} ({np.std(T)}) s")

if __name__ == "__main__":
    main()
    
