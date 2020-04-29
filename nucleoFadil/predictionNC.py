#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:37:02 2020

@author: fadiqbal
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'




file = 'input/190924_50NcLiving_B1-HT.tif'

tiffArray = io.imread(file, as_gray=False, plugin=None) 

frameNum = tiffArray.shape[0]
fHeight = tiffArray.shape[1]
fWidth = tiffArray.shape[2]



imArray = np.zeros(shape=(fHeight,fWidth))
conArray = np.zeros(tiffArray.shape, dtype='uint16')


for i1 in range(frameNum):
    imArray = tiffArray[i1]
    io.imsave(('frames/%s.tif'%(i1,)),imArray)
