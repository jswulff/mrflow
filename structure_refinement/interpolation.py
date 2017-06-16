#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:00:14 2016

@author: jonas
"""

import numpy as np
from scipy import interpolate
import cv2


def interp_spline(I, xn, yn, compute_derivs=True):
    """ Perform spline interpolation of I.
    
    I is evaluated at xn, yn.
    
    Returns
    -------
    I_warped : array_like
        Warped image
    dI_warped_dx : array_like
        Derivative of warped image in x direction
    dI_warped_dy : array_like
        Derivative of warped image in y direction
    """
    
    h,w = I.shape[:2]
    xar = np.arange(w)
    yar = np.arange(h)
    
    if I.ndim > 2:
        I_warped = np.zeros_like(I)
        dI_warped_dx = np.zeros_like(I)
        dI_warped_dy = np.zeros_like(I)
        
        for d in range(I.shape[2]):
            result = interp_spline(I[:,:,d], xn, yn, compute_derivs=compute_derivs)
            if compute_derivs:
                I_warped[:,:,d] = result[0]
                dI_warped_dx[:,:,d] = result[1]
                dI_warped_dy[:,:,d] = result[2]
            else:
                I_warped[:,:,d] = result
    else:
        S = interpolate.RectBivariateSpline(yar,xar,I,kx=2,ky=2)
        I_warped = np.clip(S.ev(yn,xn),0,1.0)
        #I_warped = S.ev(yn,xn)
        if compute_derivs:
            dI_warped_dx = S.ev(yn,xn,dy=1) # Note that y and x are swapped!
            dI_warped_dy = S.ev(yn,xn,dx=1)
    
    if compute_derivs:
        return I_warped, dI_warped_dx, dI_warped_dy
    else:
        return I_warped
        
        
        
def interp_lin(I, xn, yn, compute_derivs=True):
    """ Perform linear interpolation of I.

    I is evaluated at xn, yn.
    
    Returns
    -------
    I_warped : array_like
        Warped image
    dI_warped_dx : array_like
        Derivative of warped image in x direction
    dI_warped_dy : array_like
        Derivative of warped image in y direction
    """

    I_warped = cv2.remap(I.astype('float32'),
                         xn.astype('float32'),
                         yn.astype('float32'),
                         borderMode=cv2.BORDER_REPLICATE,
                         interpolation=cv2.INTER_CUBIC)

    if compute_derivs:
        if True:
            dI_dy, dI_dx = np.gradient(I)[:2]
            dI_warped_dy = cv2.remap(dI_dy.astype('float32'),
                                xn.astype('float32'),
                                yn.astype('float32'),
                                borderMode=cv2.BORDER_REPLICATE,
                                interpolation=cv2.INTER_CUBIC)

            dI_warped_dx = cv2.remap(dI_dx.astype('float32'),
                                xn.astype('float32'),
                                yn.astype('float32'),
                                borderMode=cv2.BORDER_REPLICATE,
                                interpolation=cv2.INTER_CUBIC)
        else:
            dI_warped_dy, dI_warped_dx = np.gradient(I_warped)[:2]

        return I_warped, dI_warped_dx, dI_warped_dy

    # If we don't want to compute the derivatives
    return I_warped
