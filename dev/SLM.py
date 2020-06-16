# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 16/06/2020
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from PyQt5 import QtCore, QtWidgets
plt.switch_backend("QT5Agg")

class SLMscreen:
    def __init__(self, resX, resY):
        """
        Initializes the window to be displayed on the SLM
        :param resX: Width in pixel
        :param resY: Height in pixel
        """
        #dirty way of finding the primary screen size, could be improved
        app = QtWidgets.QApplication([])
        screen_resolution = app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        mpl.rcParams['toolbar'] = 'None'
        mpl.rcParams['savefig.pad_inches'] = 0
        self.fig = plt.figure(figsize=(resX / 100, resY / 100), frameon=False)
        self.ax = self.fig.gca()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.figManager = plt.get_current_fig_manager()
        self.figManager.window.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        self.figManager.window.move(width, 0)
        self.figManager.window.showMaximized()
    def update(self, array):
        """
        Displays the array on the SLM
        :param array: np.ndarray
        """
        self.ax.imshow(array, cmap='gray')
        self.figManager.window.showMaximized()
        plt.pause(0.5)
    def close(self):
        """
        Closes the SLM window
        :return:
        """
        plt.close(self.fig)