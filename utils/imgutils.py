#! /usr/bin/env python2

import numpy as np
import cv2

def img_as_float(I):
    """ Convert image as float.
    Just a simple replacement routine so we do not need skimage.
    """

    if (I.dtype == np.float) or (I.dtype == np.float32) or (I.dtype == np.float64):
        return I
    else:
        return I.astype('float')/255.0

def img_as_ubyte(I):
    if (I.dtype == np.float) or (I.dtype == np.float32) or (I.dtype == np.float64):
        return (I*255.0).astype('uint8')
    else:
        return I.astype('uint8')

def rgb2gray(I):
    If = img_as_float(I)
    return cv2.cvtColor(If.astype('float32'), cv2.COLOR_RGB2GRAY).astype('float64')
