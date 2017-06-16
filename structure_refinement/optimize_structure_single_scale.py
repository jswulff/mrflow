#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:57:18 2016

@author: jonas
"""

import os,sys,pdb
import time

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy import ndimage # For median

import cv2 # For erosion


sys.path.append(os.environ['MRFLOW_HOME'])

#
# Project-local imports
#
from utils import robust_functions
from utils import edge_weights
from utils import plot_debug as pd
from utils.spyder_debug import *


#
# Module-local imports
#
import construct_matrices
import interpolate_through_H



def solve(A, b, method, tol=1e-3):
    """ General sparse solver interface.

    method can be one of
    - spsolve_umfpack_mmd_ata
    - spsolve_umfpack_colamd
    - spsolve_superlu_mmd_ata
    - spsolve_superlu_colamd
    - bicg
    - bicgstab
    - cg
    - cgs
    - gmres
    - lgmres
    - minres
    - qmr
    - lsqr
    - lsmr
    """

    if method == 'spsolve_umfpack_mmd_ata':
        return spla.spsolve(A,b,use_umfpack=True, permc_spec='MMD_ATA')
    elif method == 'spsolve_umfpack_colamd':
        return spla.spsolve(A,b,use_umfpack=True, permc_spec='COLAMD')
    elif method == 'spsolve_superlu_mmd_ata':
        return spla.spsolve(A,b,use_umfpack=False, permc_spec='MMD_ATA')
    elif method == 'spsolve_superlu_colamd':
        return spla.spsolve(A,b,use_umfpack=False, permc_spec='COLAMD')
    elif method == 'bicg':
        res = spla.bicg(A,b,tol=tol)
        return res[0]
    elif method == 'bicgstab':
        res = spla.bicgstab(A,b,tol=tol)
        return res[0]
    elif method == 'cg':
        res = spla.cg(A,b,tol=tol)
        return res[0]
    elif method == 'cgs':
        res = spla.cgs(A,b,tol=tol)
        return res[0]
    elif method == 'gmres':
        res = spla.gmres(A,b,tol=tol)
        return res[0]
    elif method == 'lgmres':
        res = spla.lgmres(A,b,tol=tol)
        return res[0]
    elif method == 'minres':
        res = spla.minres(A,b,tol=tol)
        return res[0]
    elif method == 'qmr':
        res = spla.qmr(A,b,tol=tol)
        return res[0]
    elif method == 'lsqr':
        res = spla.lsqr(A,b,atol=tol,btol=tol)
        return res[0]
    elif method == 'lsmr':
        res = spla.lsmr(A,b,atol=tol,btol=tol)
        return res[0]
    else:
        raise Exception('UnknownSolverType')


def dA_to_dU(dA, A, rigidity, occlusions, H, B, mu, q):
    """Convert structure gradient to flow gradient.
    """
    h,w = A.shape[:2]
    y,x = np.mgrid[:h,:w]    

    #
    # This could also be pre-computed
    #
    # Precompute t1,t2 = (q-x) / ||q-x||
    vx = q[0]-x
    vy = q[1]-y
    nrm = np.maximum(0.5, np.sqrt(vx**2 + vy**2))
    t1 = vx / nrm
    t2 = vy / nrm

    # Pre-compute n = ||q-x|| / mu
    n = np.sqrt(vx**2 + vy**2) / mu

    # Compute parallax at current structure
    r = A * B * n / (B*A/mu - 1)

    # And the flow from the parallax
    u_p1 = r * t1
    u_p2 = r * t2

    ## And warp back to new frame
    # Common denominator
    D = H[2,0] * (x+u_p1) + H[2,1] * (y+u_p2) + H[2,2]
    # Compute w1 and w2 (aka the x and y coordinates of the warped coordinates)
    w1 = (H[0,0] * (x+u_p1) + H[0,1] * (y+u_p2) + H[0,2]) / D
    w2 = (H[1,0] * (x+u_p1) + H[1,1] * (y+u_p2) + H[1,2]) / D

    ## Now to the same thing for A+dA.
    AdA = A+dA
    # Compute parallax
    r_dA = AdA * B * n / (B*(AdA)/mu - 1)

    # And the flow from the parallax
    u_p1_dA = r_dA * t1
    u_p2_dA = r_dA * t2

    ## And warp back to new frame
    # Common denominator
    D_dA = H[2,0] * (x+u_p1_dA) + H[2,1] * (y+u_p2_dA) + H[2,2]
    # Compute w1 and w2 (aka the x and y coordinates of the warped coordinates)
    w1_dA = (H[0,0] * (x+u_p1_dA) + H[0,1] * (y+u_p2_dA) + H[0,2]) / D_dA
    w2_dA = (H[1,0] * (x+u_p1_dA) + H[1,1] * (y+u_p2_dA) + H[1,2]) / D_dA

    du_postH = w1_dA-w1
    dv_postH = w2_dA-w2
    du_preH = u_p1 - u_p1_dA
    dv_preH = u_p2 - u_p2_dA
    return du_preH, dv_preH, du_postH, dv_postH


def clip_dA(dA, A, B, mu, q, threshold=0.5):
    h,w = A.shape[:2]
    y,x = np.mgrid[:h,:w]    

    #
    # This could also be pre-computed
    #
    # Precompute t1,t2 = (q-x) / ||q-x||
    vx = q[0]-x
    vy = q[1]-y

    # Pre-compute n = ||q-x|| / mu
    n = np.sqrt(vx**2 + vy**2) / mu

    # With r being the parallax in pixel, we want
    # | r (A) - r (A+dA) | < threshold.
    #

    # Max value for dA, so that r(A) - r(A+dA) = threshold
    dA_max = - ( mu**2 * threshold - 2*mu*threshold*A*B + threshold*A**2*B**2) / (threshold*A*B**2 + (-mu**2*n-mu*threshold)*B)
    # Min value, so that r(A) - r(A+dA) = -threshold
    dA_min = - ( mu**2 * threshold - 2*mu*threshold*A*B + threshold*A**2*B**2) / (threshold*A*B**2 + (mu**2*n-mu*threshold)*B)

    dA_clipped = np.maximum(dA_min, np.minimum(dA.reshape((h,w)), dA_max))

    return dA_clipped


# @profile
def computeDataTerm(A, I1, I2, rigidity, occlusions, H, B, mu, q, robust_ev, robust_deriv, min_weight, channel_weights=None):
    """ Compute data term between two images.
    
    Computes the data term and returns:
    
    Returns:
    --------
    A, b matrices (components of the linear system).
    E_data : Energy at current value for A.
    """

    # Weights of each channel of I1
    if channel_weights is None:
        channel_weights = [1.0,] * I1.shape[2] if I1.ndim > 2 else [1.0,]
        channel_weights = np.array(channel_weights)
    
    h,w = I1.shape[:2]
    y,x = np.mgrid[:h,:w]    

    #
    # This could also be pre-computed
    #
    # Precompute t1,t2 = (q-x) / ||q-x||
    # These are the unit vectors pointing towards the epipole.
    vx = q[0]-x
    vy = q[1]-y
    nrm = np.maximum(0.5, np.sqrt(vx**2 + vy**2))
    t1 = vx / nrm
    t2 = vy / nrm
    
    # Pre-compute n = ||q-x|| / mu
    n = np.sqrt(vx**2 + vy**2) / mu

    r = A * B * n / (B*A/mu - 1)
    # u_p1 and u_p2 are the flow in pixels. If the parallax r is positive,
    # the flow goes towards the epipole.
    u_p1 = r * t1
    u_p2 = r * t2
            
    # Compute dw_1/da, dw_2/da
    dr_da = - (n*B) / ((B * A/mu - 1)**2)

    #
    # tx, ty are cos(v), sin(v) respectively. So they can be used to construct the
    # oriented gradient.
    #

    # Warped coordinates
    xn = x + u_p1
    yn = y + u_p2

    # Computed warped I2 and derivatives
    I2w, mask_inside_frame, dI2w_dx, dI2w_dy = interpolate_through_H.interpolate_through_H(
        I2,
        H,
        xn,yn,
        compute_derivs=True)

    if I1.ndim > 2:
        dI2w_dx[mask_inside_frame==0,:] = 0
        dI2w_dy[mask_inside_frame==0,:] = 0
    else:
        dI2w_dx[mask_inside_frame==0] = 0
        dI2w_dy[mask_inside_frame==0] = 0


    # Average derivatives
    _, _, dI1w_dx, dI1w_dy = interpolate_through_H.interpolate_through_H(
        I1, np.eye(3), x, y, compute_derivs=True)

    dI2w_dx += dI1w_dx
    dI2w_dy += dI1w_dy

    if I1.ndim > 2:
        dI2w_dx /= (mask_inside_frame[:,:,np.newaxis]+1)
        dI2w_dy /= (mask_inside_frame[:,:,np.newaxis]+1)
    else:
        dI2w_dx /= (mask_inside_frame+1)
        dI2w_dy /= (mask_inside_frame+1)


    if I1.ndim > 2:
        # Average derivatives across colors
        cw = channel_weights[np.newaxis,np.newaxis,:]
        dI2w_dx = (dI2w_dx * cw).mean(axis=2)
        dI2w_dy = (dI2w_dy * cw).mean(axis=2)
    
        # Compute I_z (before averaging)
        Iz = ((I2w-I1) * cw).mean(axis=2)
    else:
        Iz = I2w - I1

    # Construct directional derivative
    dI2w_dphi = t1 * dI2w_dx + t2 * dI2w_dy
    
    # Remove pixels that leave the frame
    Iz[mask_inside_frame==0] = 0

    # Integrate rigidity and occlusions.
    Iz[occlusions==1] = 0
    Iz[rigidity==0] = 0

    # Compute psi_data
    psi_data = robust_deriv(Iz**2)

    # Compute A_data, b_data
    diag_data_image = psi_data * dI2w_dphi**2 * dr_da**2
    b_data_image = psi_data * Iz * dI2w_dphi * dr_da

    diag_data_image = ndimage.median_filter(diag_data_image, size=5)
    b_data_image = ndimage.median_filter(b_data_image, size=5)

    A_data = sparse.diags(diag_data_image.flatten(),shape=(h*w,h*w))
    b_data = b_data_image.flatten()


    E_data = robust_ev(Iz**2).sum()
    
    
    return A_data, b_data, E_data



# @profile
def optimizeStructureSingleScale(A_init,
        A_bwd_reference,
        A_fwd_reference,
        occlusions_bwd,
        occlusions_fwd,
        occlusions_both,
        rigidity,
        images,
        H_ar,
        B_ar,
        q_ar,
        mu_ar,
        lambda_1storder,
        lambda_2ndorder,
        lambda_consistency,
        A_scale_factor=1.0,
        N_outer=10, N_inner=1,
        params=None):

    EXCLUDE_BOTH_FROM_DATATERM = False
    EXCLUDE_BOTH_FROM_CONSISTENCY = False

    # Generate robust function
    robust_ev,robust_deriv = robust_functions.generate_lorentzian()

    # For the consistency, use an L1-like error function.
    robust_ev_consistency, robust_deriv_consistency = robust_functions.generate_charbonnier()

    # This is used in case of multi-scale processing
    afac_sq = A_scale_factor**2

    if params is not None and vars(params).has_key('variational_min_weight'):
        min_weight = params.variational_min_weight
    else:
        min_weight = 0.0

    # Store the original images
    images_original = [I.copy() for I in images]

    # If desired, convert images to grayscale
    if params.variational_dataterm_grayscale:
        for ni,I in enumerate(images):
            images[ni] = cv2.cvtColor(I.astype('float32'),cv2.COLOR_RGB2GRAY).astype('float64')*255.0


    channel_weights = [1.0,] if images[0].ndim == 2 else [1.0,]*images[0].shape[2]
    channel_weights = np.array(channel_weights)

    # Add gradients of images
    if images[0].ndim > 2:
        for j,I in enumerate(images):
            gy,gx = np.gradient(I)[:2]
            gy = cv2.GaussianBlur(gy.mean(axis=2), ksize=(3,3), sigmaX=-1)
            gx = cv2.GaussianBlur(gx.mean(axis=2), ksize=(3,3), sigmaX=-1)
            images[j] = np.dstack((I,gy,gx))
    else:
        for j,I in enumerate(images):
            gy,gx = np.gradient(I)
            gy = cv2.GaussianBlur(gy, ksize=(3,3), sigmaX=-1)
            gx = cv2.GaussianBlur(gx, ksize=(3,3), sigmaX=-1)
            images[j] = np.dstack((I,gy,gx))
    channel_weights = np.concatenate((0.01 * channel_weights, [1.0, 1.0]))


    #
    # Main optimization loop
    #
    h,w = images[1].shape[:2]
    y,x = np.mgrid[:h,:w]
    
    # Compute weights from current image
    weights_x, weights_y = edge_weights.computeWeightsLocallyNormalized(
        cv2.cvtColor(images_original[1].astype('float32'),
                         cv2.COLOR_RGB2GRAY))

    
    # Add rigidity. First erode to block out both inner and outer boundaries in the weights.
    rigidity_eroded = cv2.erode(rigidity.astype('uint8'),kernel=None)
    weights_x[rigidity_eroded==0] = 0 #1e-5
    weights_y[rigidity_eroded==0] = 0 #1e-5

    weights_x = np.maximum(weights_x, min_weight)
    weights_y = np.maximum(weights_y, min_weight)

    A = A_init.copy()

    amin = np.percentile(A,5)
    amax = np.percentile(A,95)
    a_extreme = np.maximum(abs(amin),abs(amax))
    
    energy_last = 1e9
    energies = []
    
    # Construct divergence and gradient matrices
#    M_DIV = construct_divergence_matrix((h,w))
    M_GRAD = construct_matrices.construct_gradient_matrix((h,w))
    M_DIV = -M_GRAD.T

    M_GRAD_2NDORDER = construct_matrices.construct_2ndorder_matrix((h,w))
    M_DIV_2NDORDER = M_GRAD_2NDORDER.T

    for k in range(N_outer):
        # Compute r (parallax) and resulting flow vectors
        ###################################################
        #
        # Data term
        #
        ###################################################
        if EXCLUDE_BOTH_FROM_DATATERM:
            occlusions_fwd_only = (occlusions_fwd + occlusions_both) > 0
            occlusions_bwd_only = (occlusions_bwd + occlusions_both) > 0
        else:
            occlusions_fwd_only = occlusions_fwd
            occlusions_bwd_only = occlusions_bwd

        A_data_fwd, b_data_fwd, E_data_fwd = computeDataTerm(
                A,
                images[1], images[2],
                rigidity, occlusions_fwd_only,
                H_ar[1], B_ar[1], mu_ar[1], q_ar[1],
                robust_ev, robust_deriv,
                min_weight,
                channel_weights)

        # Backward data term
        A_data_bwd, b_data_bwd, E_data_bwd = computeDataTerm(
                A * (mu_ar[0]/mu_ar[1]),
                images[1], images[0],
                rigidity, occlusions_bwd_only,
                H_ar[0], B_ar[0], mu_ar[0], q_ar[0],
                robust_ev, robust_deriv,
                min_weight,
                channel_weights)

        #
        # Re-normalize bwd parts to get into same range as A_fwd
        #
        A_data_bwd *= (mu_ar[0]/mu_ar[1])
        b_data_bwd *= (mu_ar[0]/mu_ar[1])

        if True:
            # Fix: insert BWD data term only in places where there are forward occlusions.
            D_fwd = A_data_fwd.diagonal()
            D_bwd = A_data_bwd.diagonal()
            pixel_replace = occlusions_fwd_only.ravel()==1
            D_fwd[pixel_replace] = D_bwd[pixel_replace]
            A_data_fwd.setdiag(D_fwd)
            b_data_fwd[pixel_replace] = b_data_bwd[pixel_replace]
            # Remove A_data_bwd and b_data_bwd
            A_data_bwd = sparse.dia_matrix((h*w,h*w))
            b_data_bwd = np.zeros_like(b_data_fwd)


        ###################################################
        #
        # Consistency term
        #
        ###################################################

        # Important note for this section:
        # A is always normed with respect to the forward motion
        # (i.e. A = Ahat_fwd = A' * mu_fwd)
        # therefore, when comparing A with A_reference_bwd, 
        # we need to re-normalize by mu_bwd / mu_fwd.

        A_bwd_ref = A_bwd_reference * (mu_ar[1]/mu_ar[0])
        A_fwd_ref = A_fwd_reference

        A_ref_max = np.maximum(A_fwd_ref, A_bwd_ref)
        A_ref_min = np.minimum(A_fwd_ref, A_bwd_ref)

        Iz = (A-A_ref_max) * (A>A_ref_max) + (A-A_ref_min) * (A<A_ref_min)

        # This is somewhat questionable
        Iz[occlusions_bwd==1] = (A - A_fwd_ref)[occlusions_bwd==1]
        Iz[occlusions_fwd==1] = (A - A_bwd_ref)[occlusions_fwd==1]

        if EXCLUDE_BOTH_FROM_CONSISTENCY:
            Iz[occlusions_both==1] = 0

        psi_consistency = afac_sq * robust_deriv_consistency(afac_sq * Iz**2)
        A_consistency = sparse.diags(psi_consistency.flatten(), shape=(h*w,h*w))
        b_consistency = A_consistency.dot(Iz.flatten())

        E_consistency = robust_ev_consistency(afac_sq * Iz**2).sum()
        E_consistency_fwd = E_consistency_bwd = E_consistency


        ###################################################
        #
        # 1st order term
        #
        ###################################################
        grad_A_1storder = M_GRAD.dot(A.flatten())
        grad_A_1storder_reshaped = grad_A_1storder.reshape((-1,h*w))
        weightarray_1storder = np.c_[weights_x.flatten(), weights_y.flatten()].T

        psi_smooth_1storder = afac_sq * robust_deriv(afac_sq * (weightarray_1storder * grad_A_1storder_reshaped**2).sum(axis=0))
        
        P = sparse.diags(
                np.r_[ (weights_x.flatten()*psi_smooth_1storder).flatten(),
                       (weights_y.flatten()*psi_smooth_1storder).flatten()])
        A_smooth_1storder = -M_DIV.dot(P).dot(M_GRAD)
        b_smooth_1storder = A_smooth_1storder * A.flatten()


        ###################################################
        #
        # 2nd order term
        #
        ###################################################
        
        # In this term, set the weights actually to zero
        weights_x[rigidity_eroded==0] = min_weight
        weights_y[rigidity_eroded==0] = min_weight

        # Somewhat hacky.
        weights_xy = np.minimum(weights_x,weights_y)

        # This contains the 2nd derivative images, vectorized and stacked
        grad_A_2ndorder = M_GRAD_2NDORDER.dot(A.flatten())

        weightarray_2ndorder = np.c_[weights_x.flatten(),weights_y.flatten()].T

        # The components are summed within the robust function, so we first
        # need to reshape grad_A_2ndorder, so that it contains the vectorized
        # derivative images as rows.
        grad_A_2ndorder_reshaped = grad_A_2ndorder.reshape((-1,h*w))
        psi_smooth_2ndorder = afac_sq * robust_deriv( afac_sq * (weightarray_2ndorder * grad_A_2ndorder_reshaped**2).sum(axis=0) )

        P = sparse.diags(np.r_[
            psi_smooth_2ndorder.flatten() * weights_x.flatten(),
            psi_smooth_2ndorder.flatten() * weights_y.flatten()])


        A_smooth_2ndorder = M_DIV_2NDORDER.dot(P).dot(M_GRAD_2NDORDER)
        b_smooth_2ndorder = A_smooth_2ndorder * A.flatten()

        #
        #
        # Compute energy
        #
        #
        # Compute energy
        
        E_1storder = robust_ev(afac_sq * (weightarray_1storder * grad_A_1storder_reshaped**2).sum(axis=0)).sum()
        E_2ndorder = robust_ev(afac_sq * (weightarray_2ndorder * grad_A_2ndorder_reshaped**2).sum(axis=0)).sum()

        
        energy = E_data_fwd + E_data_bwd + lambda_1storder * E_1storder + lambda_2ndorder * E_2ndorder + lambda_consistency * E_consistency
        
        print('({0:02d}) EpPx={1:2.5f} -- Data=({2:2.1f}/{3:2.1f}) -- Consistency=({4:2.1f}/{5:2.1f}) -- 1st ord={6:2.1f} -- 2nd ord={7:2.1f}'.format(
              k,energy/(h*w),
              E_data_bwd, E_data_fwd,
              lambda_consistency*E_consistency_bwd,
              lambda_consistency*E_consistency_fwd,
              lambda_1storder * E_1storder,
              lambda_2ndorder * E_2ndorder))
        energies.append(energy/(h*w))
            
        
        # Solve system -> obtain da

        # Use both forward and backward data term
        A_combined = A_data_fwd + A_data_bwd + lambda_1storder * A_smooth_1storder + lambda_2ndorder * A_smooth_2ndorder + lambda_consistency * A_consistency
        b_combined = -b_data_fwd - b_data_bwd - lambda_1storder * b_smooth_1storder - lambda_2ndorder * b_smooth_2ndorder - lambda_consistency * b_consistency


        # Add unit matrix to maybe improve conditioning?
        # Build matrix to zero out the rigidity parts
        z = sparse.diags((rigidity==1).ravel().astype('float32'),shape=(h*w,h*w))
        A_combined = z.dot(A_combined).dot(z)
        b_combined = z.dot(b_combined)


        diag_cond = np.sign(A_combined.diagonal())
        diag_cond[:] = 1
        diag_cond[diag_cond==0] = 1
        diag_cond *= 0.01
        A_combined += sparse.diags(diag_cond).tocsr()

 
        t0 = time.time()
        da = solve(A_combined, b_combined, method='minres', tol=1e-5)
        t1 = time.time()

        print('({0:02d}) \t Solving took {1} seconds for sequence {2}'.format(k, t1-t0, params.tempdir))


        # Clipping of da -- this is a bit questionable.
        if np.any(np.isnan(da)):
            print('  (EE) {} of {} entries are NaN. Setting to zero...'.format(np.isnan(da).sum(), da.size))
            da[np.isnan(da)] = 0

        if np.any(np.isinf(da)):
            print('  (EE) {} of {} entries are INF. Setting to zero...'.format(np.isinf(da).sum(), da.size))
            da[np.isinf(da)] = 0

        print('  (MM) Max of da={}\tMin of da={}'.format(da.max(),da.min()))
        print('  (MM) Clipping off {} values'.format((da>1).sum() + (da<-1).sum()))

        da = clip_dA(da, A, B_ar[1], mu_ar[1], q_ar[1],threshold=1.0)
        da = np.clip(da,-1.0,1.0)


        if params.debug_save_frames:
            damin = abs(np.percentile(da,5))
            damax = abs(np.percentile(da,95))
            da_extreme = max(damin,damax)
            # da_extreme = 0.1
            pd.plot_image(da.reshape(A.shape), '[05 Optimization Variational] Gradient before median, iteration {}'.format(k), vmin=-da_extreme,vmax=da_extreme,outpath=params.tempdir,colorbar=True)


        da = da.reshape(A.shape)

        ## Compute du,dv from dA.
        du_preh,dv_preh, du_posth, dv_posth = dA_to_dU(da, A, rigidity, occlusions_fwd_only, H_ar[1], B_ar[1], mu_ar[1], q_ar[1])
        dUV_preH = np.sqrt(du_preh**2 + dv_preh**2)
        dUV_postH = np.sqrt(du_posth**2 + dv_posth**2)
        print('  (MM) dUV_preH min={}\tmax={}'.format(dUV_preH.min(), dUV_preH.max()))
        print('  (MM) dUV_postH min={}\tmax={}'.format(dUV_postH.min(), dUV_postH.max()))



        # Median filtering of da

        da = ndimage.median_filter(A + da, size=7) - A

        if params.debug_save_frames:
            damin = abs(np.percentile(da,5))
            damax = abs(np.percentile(da,95))
            da_extreme = max(damin,damax)
            da_extreme = 0.003
            pd.plot_image(da.reshape(A.shape), '[05 Optimization Variational] Gradient after median, iteration {}'.format(k), vmin=-da_extreme,vmax=da_extreme,outpath=params.tempdir,colorbar=True)
            if k==0:
                pd.plot_image(A, '[05 Optimization Variational] Structure, iteration {}'.format(k), vmin=-a_extreme,vmax=a_extreme,outpath=params.tempdir,colorbar=True)

            pd.plot_image(A+da, '[05 Optimization Variational] Structure, iteration {}'.format(k+1), vmin=-a_extreme,vmax=a_extreme,outpath=params.tempdir,colorbar=True)

        #print('Updating...')
        A = A + da

    return A, energies
