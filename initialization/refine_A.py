#! /usr/bin/env python2

"""
Refine structure, homography, camera motion so that the backward structure
best matches the forward structure.
"""

import time
import numpy as np
import chumpy as ch
from scipy import sparse
from scipy import optimize

from utils.spyder_debug import *

import chumpy as ch

def chumpy_get_H(p1,p2):
    """ Compute differentiable homography from p1 to p2.
    
    Parameters
    ----------
    p1,p2 : array_like, shape (4,2)
        Containing points to match.
    """
    xmin = 0
    ymin = 0
    xmax = 1024
    ymax = 576
    p1 = 2 * p1 / ch.array([[xmax,ymax]])
    p1 = p1 - 1.0
    p2 = 2 * p2 / ch.array([[xmax,ymax]])
    p2 = p2 - 1.0

    N = p1.shape[0]
    A1 = ch.vstack((
        ch.zeros((3, N)),
        -p1.T,
        -ch.ones((1, N)),
        p2[:,1] * p1[:,0],
        p2[:,1] * p1[:,1],
        p2[:,1] )).T
    A2 = ch.vstack(( 
        p1.T, ch.ones((1,N)),
        ch.zeros((3, N)),
        -p2[:,0] * p1[:,0],
        -p2[:,0] * p1[:,1],
        -p2[:,0] )).T
    A = ch.vstack((A1,A2))

    U,S,V = ch.linalg.svd(A.T.dot(A))
    H_new = V[-1,:].reshape((3,3))

    # Re-normalize
    ML = ch.array([[xmax/2.0, 0.0, xmax/2.0],[0,ymax/2.0,ymax/2.0],[0,0,1.0]])
    MR = ch.array([[2.0/xmax, 0.0, -1.0], [0, 2.0/ymax, -1.0],[0, 0, 1.0]])
    H_new = ML.dot(H_new).dot(MR)

    return H_new / H_new[2,2]

def compute_cost(params, p0, u_fwd, v_fwd, A_reference, mu_reference, mu_refine, x, y, sigma=1.0, optimize_H=True, optimize_q=False, optimize_B=True):
    """ Compute structure matching cost using chumpy
    
    Parameters:
    x[0] - x[7]: Coordinates of corner points in second frame
    x[8],x[9]: Q
    x[10]: B
    
    p1 : Coordinates of corner points in first frame
    
    We optimize over all parameters
    """
    
    params_ch = ch.array(params)
    
    
    # Extract corner points
    p1 = params_ch[:8].reshape((4,2))
    q = params_ch[8:10]
    B = params_ch[10]

    # Convert the ones we do not want to 
    if not optimize_B:
        B = B()
    if not optimize_H:
        p1 = np.array(p1)
    if not optimize_q:
        q = np.array(q)

    # Compute unity vectors towards q
    vx = q[0] - x
    vy = q[1] - y
    nrm = ch.maximum(1.0,ch.sqrt((vx**2 + vy**2)))
    vx /= nrm
    vy /= nrm

    if not optimize_q:
        vx = np.copy(vx)
        vy = np.copy(vy)

    # Compute differentiable homography (mapping I1 to I0)
    H_inv = chumpy_get_H(p1,p0)
    if not optimize_H:
        H_inv = np.copy(H_inv)


    # Remove differentiable homography from backward flow
    x_new = x + u_fwd
    y_new = y + v_fwd
    
    D = H_inv[2,0] * x_new + H_inv[2,1] * y_new + H_inv[2,2]
    u_parallax = (H_inv[0,0] * x_new + H_inv[0,1] * y_new + H_inv[0,2]) / D - x
    v_parallax = (H_inv[1,0] * x_new + H_inv[1,1] * y_new + H_inv[1,2]) / D - y
    r = u_parallax * vx + v_parallax * vy
    
    # Compute A estimates
    A_refined = r / (B * (r / mu_refine - nrm / mu_refine))
    
    # The data we want to match
    z_match = A_reference * (mu_refine/mu_reference)
    z = A_refined

    #
    # Use Geman-Mcclure error
    #

    err_per_px = (z-z_match)**2
    err = (sigma*err_per_px / (sigma**2 + err_per_px)).sum()
    # Lorentzian
    # err = (sigma * ch.log(1+0.5 * err_per_px/(sigma**2))).sum()
    derr = err.dr_wrt(params_ch)

    return err,np.copy(derr).flatten()


def dlt(p1,p2):
    """Compute homography based on the Direct Linear Transform.

    No normalization is performed, so p1 and p2 should be
    well-distributed (for example, corresponding to the corners
    of the image).
    """
    N = p1.shape[0]
    A1 = np.vstack(( np.zeros((3, N)), -p1.T, -np.ones((1, N)), p2[:,1] * p1[:,0], p2[:,1] * p1[:,1], p2[:,1] )).T
    A2 = np.vstack(( p1.T, np.ones((1,N)), np.zeros((3, N)), -p2[:,0] * p1[:,0], -p2[:,0] * p1[:,1], -p2[:,0] )).T
    A = np.vstack((A1,A2))

    U,S,V = np.linalg.svd(A.T.dot(A))
    H_new = V[-1,:].reshape((3,3))    
    return H_new / H_new[2,2]



def compute_A_simple(H, q, B, mu, u, v):
    """ Direct, non-differentiable function to compute A
    """
    y,x = np.mgrid[:u.shape[0],:u.shape[1]]

    Hinv = np.linalg.inv(H)
    # Compute unity vectors towards q
    vx = q[0] - x
    vy = q[1] - y
    nrm = np.maximum(1.0,np.sqrt((vx**2 + vy**2)))
    vx /= nrm
    vy /= nrm


    # Remove differentiable homography from backward flow
    x_new = x + u
    y_new = y + v
    D = Hinv[2,0] * x_new + Hinv[2,1] * y_new + Hinv[2,2]
    u_parallax = (Hinv[0,0] * x_new + Hinv[0,1] * y_new + Hinv[0,2]) / D - x
    v_parallax = (Hinv[1,0] * x_new + Hinv[1,1] * y_new + Hinv[1,2]) / D - y

    # Project onto epilines
    r = u_parallax * vx + v_parallax * vy

    # Compute new A
    A = r / (B*(r / mu - nrm / mu))
    
    return A


def compute_sigma(p0, p1, q, B, mu, u, v, A_reference, mu_reference,mask):
    #
    # Compute structure difference to obtain sigma in robust func
    #
    H_new_tmp = dlt(p0,p1.reshape((4,2)))
    A_new_tmp = compute_A_simple(H_new_tmp,
                                 q,
                                 B,
                                 mu,
                                 u,
                                 v)

    # Compute A estimates
    err = np.abs(A_new_tmp - A_reference * (mu/mu_reference))

    # Explicitly exclude outliers
    mask_current = np.zeros_like(A_reference)>0
    mask_current[err <= np.median(err)] = True
    # mask_current *= mask

    err = (A_new_tmp - A_reference * (mu/mu_reference))[mask]
    mad = np.median(np.abs(err-np.median(err))) * 1.426
    sigma = mad / 1.426

    print('--- sigma = {} ---'.format(sigma))
    print('--- Median error = {} ---'.format(np.median(err)))
    print('--- Median abs error = {} ---'.format(np.median(abs(err))))

    return sigma, mask_current



def refine_A(u_refine,
             v_refine,
             H_refine,
             B_refine,
             q_refine, # Not refined at this point
             mu_refine,
             mu_reference,
             A_reference,
             rigidity,
             no_occlusions, # Binary map denoting areas that are valid in both frames.
             refine_B=True, # Flag to see if B should be refined. For the forward pass, ignore B.
):
    """ Refine homography and camera motion to minimize structure error.

    Refine homography and camera motion so that the resulting structure
    best matches the given reference structure.

    A robust function (median) is used, and only pixels within the rigidity
    mask are considered.
    """

    h,w = A_reference.shape
    y,x = np.mgrid[:h,:w]

    # Define corner points to parameterize the homography.
    # p0 are the points in the reference frame (fixed), p1 in the refinement frame.
    # We refine the homography mapping p0 to p1 (H_refine maps p0 to p1)
    p0 = np.array([[0,0.0], [0.0,h],[w,0.0],[w,h]])
    # Last line of homography
    D = H_refine[2,0] * p0[:,0] + H_refine[2,1] * p0[:,1] + H_refine[2,2]
    p1_x = (H_refine[0,0] * p0[:,0] + H_refine[0,1] * p0[:,1] + H_refine[0,2]) / D
    p1_y = (H_refine[1,0] * p0[:,0] + H_refine[1,1] * p0[:,1] + H_refine[1,2]) / D
    p1 = np.c_[p1_x,p1_y]

    p1_iter = p1.copy()
    B_refine_iter = B_refine
    q_refine_iter = q_refine

    for scale in [16,8,4,2]:

        # Some optimization parameters
        x0 = np.hstack((p1_iter.flatten(), q_refine_iter, B_refine_iter))
        if scale==16:
            maxiter=1000
        else:
            maxiter=scale*5
        ftol=1e-6


        # Get valid pixels at current sampling
        mask = np.zeros_like(rigidity)
        mask[::scale,::scale] = 1
        mask *= (rigidity>0)
        mask *= (no_occlusions>0)

        # To compute sigma, exclude occlusiosn and rigidity, but use full frame
        mask_sigma = (no_occlusions>0) * (rigidity>0)

        #
        # Step 1: Optimize Q
        #
        sigma, mask_current = compute_sigma(p0, p1_iter,
                                            q_refine_iter,
                                            B_refine_iter,
                                            mu_refine,
                                            u_refine,
                                            v_refine,
                                            A_reference,
                                            mu_reference,
                                            mask_sigma)
        mask_current = mask_current*mask

        args = (p0, 
                u_refine[mask_current],
                v_refine[mask_current],
                A_reference[mask_current],
                mu_reference, mu_refine,
                x[mask_current], y[mask_current], sigma,
                False, #Optimize_H
                True, #Optimize_q
                False, #Optimize_B
                )

        t0=time.time()
        res = optimize.minimize(
                compute_cost,
                jac=True,
                x0=x0,
                args=args,
                method='L-BFGS-B',
                options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})
        t1=time.time()
        print('\tScale {}, Q: Optimization took {} seconds'.format(scale,t1-t0))
        print('\t\t q: {} => {}'.format(x0[8:10],res.x[8:10]))
        q_refine_iter = res.x[8:10]

        # Save as new starting point
        x0[8:10] = q_refine_iter

        #
        # Step 2: Optimize H and B
        #
        sigma, mask_current = compute_sigma(p0, p1_iter,
                                            q_refine_iter,
                                            B_refine_iter,
                                            mu_refine,
                                            u_refine,
                                            v_refine,
                                            A_reference,
                                            mu_reference,
                                            mask_sigma)
        mask_current = mask_current*mask

        args = (p0, 
                u_refine[mask_current],
                v_refine[mask_current],
                A_reference[mask_current],
                mu_reference, mu_refine,
                x[mask_current], y[mask_current], sigma,
                True, #Optimize_H
                False, #Optimize_q
                True, #Optimize_B
                )

        t0=time.time()
        res = optimize.minimize(
                compute_cost,
                jac=True,
                x0=x0,
                args=args,
                method='L-BFGS-B',
                options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})
        t1=time.time()
        print('\tScale {}: Optimization took {} seconds'.format(scale,t1-t0))
        p1_iter = res.x[:8]
        B_refine_iter = res.x[10]


    # Now p0_iter and B_refine iter hold the best new estimates.

    H_new = dlt(p0,p1_iter.reshape((4,2)))
    A_new = compute_A_simple(H_new,
            q_refine_iter,
            B_refine_iter,
            mu_refine,
            u_refine,
            v_refine)

    return A_new, H_new, B_refine_iter, q_refine_iter


