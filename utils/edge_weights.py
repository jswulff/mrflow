#! /usr/bin/env python2

import numpy as np
import cv2

def computeWeightsGloballyNormalized(I):
    if I.ndim == 2:
        I = I[:,:,np.newaxis]

    h,w,c = I.shape

    diff_0 = np.vstack(( np.diff(I, axis=0).sum(axis=2) / c, np.zeros((1,w)) ))
    diff_1 = np.hstack(( np.diff(I, axis=1).sum(axis=2) / c, np.zeros((h,1)) ))

    diff_ne = np.zeros((h,w))
    diff_ne[1:,:-1] = (I[1:,:-1,:] - I[:-1,1:,:]).sum(axis=2) / c

    diff_se = np.zeros((h,w))
    diff_se[:-1,:-1] = (I[:-1,:-1,:] - I[1:,1:,:]).sum(axis=2)/c

    beta_0 = 1.0 / ( 2 * max(1e-6, (diff_0**2).mean()) )
    beta_1 = 1.0 / ( 2 * max(1e-6, (diff_1**2).mean()) )

    beta_ne = 1.0 / (2 * max(1e-6, (diff_ne**2).mean()))
    beta_se = 1.0 / (2 * max(1e-6, (diff_se**2).mean()))

    w_vertical = np.exp( - diff_0**2 * beta_0).astype('float32')
    w_horizontal = np.exp( - diff_1**2 * beta_1).astype('float32')

    w_ne = np.exp( - diff_ne**2 * beta_ne).astype('float32')
    w_se = np.exp( - diff_se**2 * beta_se).astype('float32')

    return w_horizontal, w_vertical, w_ne, w_se




def computeWeightsLocallyNormalized(I, centered_gradient=True, norm_radius=45):
    h,w = I.shape[:2]

    if centered_gradient:
        gy,gx = np.gradient(I)[:2]
        gysq = (gy**2).mean(axis=2) if gy.ndim > 2 else gy**2
        gxsq = (gx**2).mean(axis=2) if gx.ndim > 2 else gx**2

        gxsq_local_mean = cv2.blur(gxsq, ksize=(norm_radius, norm_radius))
        gysq_local_mean = cv2.blur(gysq, ksize=(norm_radius, norm_radius))

        w_horizontal = np.exp( - gxsq * 1.0/(2*np.maximum(1e-6, gxsq_local_mean)))
        w_vertical = np.exp( - gysq * 1.0/(2*np.maximum(1e-6, gysq_local_mean)))

    else:
        raise Exception("NotImplementedYet")

    return w_horizontal, w_vertical


