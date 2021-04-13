# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 16/06/2020
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
plt.switch_backend("QT5Agg")


class SLMscreen:

    def __init__(self, resX, resY):
        plt.ioff()
        """
        Initializes the window to be displayed on the SLM
        :param resX: Width in pixel
        :param resY: Height in pixel
        """
        # dirty way of finding the primary screen size, could be improved
        app = QApplication([])
        screen_resolution = app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        mpl.rcParams['toolbar'] = 'None'
        mpl.rcParams['image.interpolation'] = 'None'
        mpl.rcParams['image.resample'] = False
        self.fig = plt.figure(0, figsize=(resX / 100, resY / 100), frameon=False)
        self.ax = plt.axes([0, 0, 1, 1], frameon=False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.im = self.ax.imshow(np.zeros((resY, resX), dtype='uint8'), cmap='gray', aspect='equal', vmin=0, vmax=255)
        self.window = self.fig.canvas.window()
        self.window.setWindowFlags(Qt.FramelessWindowHint | Qt.CustomizeWindowHint)
        self.window.setGeometry(width, 0, resX, resY)
        self.window.statusBar().setVisible(False)
        self.window.showMaximized()
        #this is ugly but else you need to update it twice before it works fine
        self.update(np.zeros((resY, resX), dtype='uint8'))
        self.update(np.zeros((resY, resX), dtype='uint8'))

    def update(self, array, sleep=0.01):
        """
        Displays the array on the SLM
        :param array: np.ndarray
        """
        self.fig.canvas.flush_events()
        self.im.set_data(array)
        self.fig.canvas.draw()
        self.window.showMaximized()
        time.sleep(sleep)

    def close(self):
        """
        Closes the SLM window and resets the matplotlib rcParams parameters
        :return:
        """
        plt.close(self.fig)
        mpl.rcParams['toolbar'] = 'toolbar2'
        mpl.rcParams['image.interpolation'] = 'None'
        mpl.rcParams['image.resample'] = False
