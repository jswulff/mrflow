#! /usr/bin/env python2

import numpy as np
import cv2
import interpolation

def interpolate_through_H(I, H, xn, yn, compute_derivs=True, dIdx=None, dIdy=None):
    """ Compute interpolation after an additional homography step.

    Compute the values J(x,y) = I( H( xn(x,y), yn(x,y) ) ).

    If compute_derivs is True, also compute the derivatives
    dJ/dx and dJ/dy.

    Returns
    -------
    J : array_like
        Warped image
    mask : array_like
        Mask that is 1 for all pixels inside the frame and 0 otherwise.
    dJ_dx : array_like
        Derivative of warped image in x direction
    dJ_dy : array_like
        Derivative of warped image in y direction
    
    """

    H_ = H.astype('float')
    xn_ = xn.astype('float')
    yn_ = yn.astype('float')

    # Delimiter
    DH = xn * H[2,0] + yn * H[2,1] + H[2,2]
    xnH = (xn * H[0,0] + yn * H[0,1] + H[0,2]) / DH
    ynH = (xn * H[1,0] + yn * H[1,1] + H[1,2]) / DH

    # Build mask
    h,w = I.shape[:2]
    mask = (xnH >=0) * (xnH <= w-1) * (ynH >=0) * (ynH <= h-1)
    mask = mask.astype('float')

    J = interpolation.interp_lin(I, xnH, ynH, compute_derivs=False)

    if compute_derivs:
        if dIdx is None or dIdy is None:
            dIdx, dIdy = compute_derivs_through_H(I, H)
        dJdy = interpolation.interp_lin(dIdy, xnH, ynH, compute_derivs=False)
        dJdx = interpolation.interp_lin(dIdx, xnH, ynH, compute_derivs=False)
        return J, mask, dJdx, dJdy

    return J, mask

def compute_derivs_through_H(I, H):
    """ Compute derivatives after additional homography step.

    For J(x,y) = I( H (x,y) ), compute dJ/dx and dJ/dy.
    """

    h,w = I.shape[:2]

    Hinv = np.linalg.inv(H)
    H_xp1_only = np.array([[1.0,0,-1.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    H_xm1_only = np.array([[1.0,0,1.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    H_yp1_only = np.array([[1.0, 0, 0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    H_ym1_only = np.array([[1.0, 0, 0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    H_only_array = [H_xp1_only, H_xm1_only, H_yp1_only, H_ym1_only]
    [H_xp1, H_xm1, H_yp1, H_ym1] = [Hinv.dot(H_).dot(H) for H_ in H_only_array]

    y,x = np.mgrid[:h,:w]
    xy_ar = np.c_[x.flatten(), y.flatten()].astype('float')

    # Compute displacements of finite difference samples.
    xy_ar_xp1 = ptransform(xy_ar, H_xp1)
    xy_ar_xm1 = ptransform(xy_ar, H_xm1)
    xy_ar_yp1 = ptransform(xy_ar, H_yp1)
    xy_ar_ym1 = ptransform(xy_ar, H_ym1)

    d_xp1 = np.linalg.norm(xy_ar - xy_ar_xp1, axis=1).reshape((h,w))
    d_xm1 = np.linalg.norm(xy_ar - xy_ar_xm1, axis=1).reshape((h,w))
    d_yp1 = np.linalg.norm(xy_ar - xy_ar_yp1, axis=1).reshape((h,w))
    d_ym1 = np.linalg.norm(xy_ar - xy_ar_ym1, axis=1).reshape((h,w))

    if I.ndim > 2:
        d_xp1 = d_xp1[:,:,np.newaxis]
        d_xm1 = d_xm1[:,:,np.newaxis]
        d_yp1 = d_yp1[:,:,np.newaxis]
        d_ym1 = d_ym1[:,:,np.newaxis]

    I_xp1 = cv2.warpPerspective(I, H_xp1, (w,h), borderMode=cv2.BORDER_REPLICATE)
    I_xm1 = cv2.warpPerspective(I, H_xm1, (w,h), borderMode=cv2.BORDER_REPLICATE)
    I_yp1 = cv2.warpPerspective(I, H_yp1, (w,h), borderMode=cv2.BORDER_REPLICATE)
    I_ym1 = cv2.warpPerspective(I, H_ym1, (w,h), borderMode=cv2.BORDER_REPLICATE)

    dx = 0.5 * ((I_xp1 - I) * d_xp1 + (I - I_xm1) * d_xm1)
    dy = 0.5 * ((I_yp1 - I) * d_yp1 + (I - I_ym1) * d_ym1)
    return dx,dy


def ptransform(xy, M):
    """ Just a small wrapper to cv2.perspectiveTransform
    """
    if xy.shape[1] == 2:
        return cv2.perspectiveTransform(xy[:,np.newaxis,:], M).squeeze()
    else:
        return cv2.perspectiveTransform(xy.T[:,np.newaxis,:], M).squeeze().T
