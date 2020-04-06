# -*- coding: utf-8 -*-
"""
@author : Tangui ALADJIDI
"""
import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *
from PIL import Image #for custom phase / intensity masks

class Field:
    def __init__(self):
        self.Field = Begin()
        self.Intensity =
        #TODO
        #should inherit the Field class from lightpipes more or less in a 2D array format.
class Setup:
    def __init__(self, **kwargs):
        """
        Initiates the class
        :param kwargs: "f" for a lens of focal length f. "I0" for initial intensity. "phi0" for initial phase. etc ...
        """
        #physical and numerical params for the simulation
        self.size=4*cm
        self.wavelength=781*nm
        self.N=2048
        self.field = Begin(size, wavelength, N)
        self.elements = []
        #optional arguments
        #TODO
        """
            self.f=100*cm #focal length
            #initiate custom phase and intensity filters emulating the SLM
            phi0 = np.asarray(Image.open("harambe.bmp"))[:,:,2] #extract only the first channel
            #phi0 = np.asarray(Image.open("calib.bmp"))
            #print(np.max(phi0)
            phi0= (phi0+128)*(2*np.pi/255) #conversion to rads
            #apply SLM filter to initiate the field in the SLM plane
            Field=RectAperture(1*cm,1*cm,0,0,0,Field)
            Field=SubPhase(phi0,Field)
        """
    def add_element(self, element: str, **kwargs):
        """
        Adds optical elements to the setup
        :param element: Name of the element. Must be a LightPipes element for now.
        :param kwargs: keywords arguments corresponding to each element. Example : "f" for a lens of focal length f.
        :return: None
        """
        #TODO
        pass
    def get_image_field(self):
        #TODO
        pass
    def propagate(self, z: float):
        """
        Propagates the field in the setup for a given distance
        :param z: Propagation distance
        :return: The propagated field, intensity and amplitude
        """
        #TODO
        pass
    def phase_retriever(self, I0: np.ndarray, I: np.ndarray, f: float, N: int, N_mod: int):
        """
        Retrieves the initial phase given the image plane of the setup
        :param I0: Source intensity field
        :param I: Intensity field from which to retrieve the phase
        :param f: Focal length of the lens conjugating the two planes
        :param N: Number of iterations for GS algorithm
        :return: The calculated phase_map using Gerchberg-Saxton algorithm
        """
