#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:57:18 2016

@author: jonas
"""

import os,sys,pdb

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import ndimage # For median


sys.path.append(os.environ['PPPFLOW_HOME'])

#
# Project-local imports
#
from utils import robust_functions
from utils import edge_weights
from utils.spyder_debug import *

#
# Module-local imports
#
from interpolation import interp_spline
import construct_matrices



def optimizeStructureSingleScale(A_init,
                                 A_reference, # Reference init to lock consistency to
                                 I1, I2,
                                 H, B, q, mu,
                                 lambda_1storder,
                                 lambda_2ndorder,
                                 lambda_consistency,
                                 N_outer=10, N_inner=1):
    
    # Generate robust function
    robust_ev,robust_deriv = robust_functions.generate_lorentzian()

    #
    # Main optimization loop
    #
    h,w = I1.shape[:2]
    y,x = np.mgrid[:h,:w]    
    
    # Precompute t1,t2 = (q-x) / ||q-x||
    vx = q[0]-x
    vy = q[1]-y
    nrm = np.maximum(0.5, np.sqrt(vx**2 + vy**2))
    t1 = vx / nrm
    t2 = vy / nrm
    
    # Pre-compute n = ||q-x|| / mu
    n = np.sqrt(vx**2 + vy**2) / mu

    # Compute weights from current image
    weights_x, weights_y = edge_weights.computeWeightsLocallyNormalized(I1)
#    weights_x, weights_y = edge_weights.computeWeightsGloballyNormalized(I1)[:2]
    
    
    A = A_init.copy()
    
    energy_last = 1e9
    energies = []
    
    # Construct divergence and gradient matrices
#    M_DIV = construct_divergence_matrix((h,w))
    M_GRAD = construct_matrices.construct_gradient_matrix((h,w))
    M_DIV = -M_GRAD.T
    
    M_GRAD_2NDORDER = construct_matrices.construct_2ndorder_matrix((h,w))
    M_DIV_2NDORDER = -M_GRAD_2NDORDER.T
    
    
    
    for k in range(N_outer):
        # Compute r (parallax) and resulting flow vectors
        r = A * n / (A/mu - B)
        u_p1 = r * t1
        u_p2 = r * t2
                
        # Compute dw_1/da, dw_2/da
        D = H[2,0] * (x+u_p1) + H[2,1] * (y+u_p2) + H[2,2]
        dr_da = - (n*B) / ((A/mu - B)**2)
        dw1_da = 1.0 / (D**2) * (D * (t1 * H[0,0] + t2 * H[0,1]) - (H[0,0] * (x+u_p1) + H[0,1] * (y+u_p2) + H[0,2]) * (t1 * H[2,0] + t2 * H[2,1])) * dr_da
        dw2_da = 1.0 / (D**2) * (D * (t1 * H[1,0] + t2 * H[1,1]) - (H[1,0] * (x+u_p1) + H[1,1] * (y+u_p2) + H[1,2]) * (t1 * H[2,0] + t2 * H[2,1])) * dr_da
        
        # Compute w1 and w2 (aka the x and y coordinates of the warped coordinates)
        w1 = (H[0,0] * (x+u_p1) + H[0,1] * (y+u_p2) + H[0,2]) / D
        w2 = (H[1,0] * (x+u_p1) + H[1,1] * (y+u_p2) + H[1,2]) / D
        
        
        # Compute I2(w) and its derivatives
        I2w, dI2w_dx, dI2w_dy = interp_spline(I2, w1, w2)
        
        if I1.ndim > 2:
            # Average derivatives across colors
            dI2w_dx = dI2w_dx.mean(axis=2)
            dI2w_dy = dI2w_dy.mean(axis=2)
        
            # Compute I_z (before averaging)
            #Iz = np.sqrt(((I2w - I1)**2).sum(axis=2))
            Iz = I2w.mean(axis=2) - I1.mean(axis=2)
        else:
            Iz = I2w - I1
        
        # Remove pixels that leave the frame
        in_frame = (w1 >= 0) * (w1 < w) * (w2 >= 0) * (w2 < h)
        Iz[in_frame==0] = 0

       
        da = 0
        for l in range(N_inner):
            
            if N_inner == 1:
                # In this case, we do not have to recompute Iz_bar and A_l.
                A_l = A
                Iz_bar = Iz
                
            else:
                # Compute I_z_bar (non-discretized)
                A_l = A + da
                r_l = A_l * n / (A_l/mu - B)
                u_p1_l = r_l * t1
                u_p2_l = r_l * t2
                D_l = H[2,0] * (x+u_p1_l) + H[2,1] * (y+u_p2_l) + H[2,2]
                w1_l = (H[0,0] * (x+u_p1_l) + H[0,1] * (y+u_p2_l) + H[0,2]) / D_l
                w2_l = (H[1,0] * (x+u_p1_l) + H[1,1] * (y+u_p2_l) + H[1,2]) / D_l
                I2w_l = interp_spline(I2,w1_l,w2_l,compute_derivs=False)
                
                #Iz_bar = (I2w_l - I1).mean(axis=2)
                Iz_bar = np.sqrt(((I2w_l - I1)**2).sum(axis=2))
                
                
                # Remove pixels that leave the frame
                in_frame = (w1_l >= 0) * (w1_l < w) * (w2_l >= 0) * (w2_l < h)
                Iz_bar[in_frame==0] = 0
            
    
            ###################################################
            #
            # Data term
            #
            ###################################################
            # Compute psi_data
            psi_data = robust_deriv(Iz_bar**2)
            
            # Compute A_data, b_data
            diag_data = psi_data * ((dI2w_dx * dw1_da + dI2w_dy * dw2_da)**2)
            A_data = sparse.diags(diag_data.flatten(),shape=(h*w,h*w))
            b_data = (psi_data * Iz * (dI2w_dx * dw1_da + dI2w_dy * dw2_da)).flatten()
            
            
            
            ###################################################
            #
            # Consistency term
            #
            ###################################################
            psi_consistency = robust_deriv((A_l - A_reference)**2)

            # Compute A_data, b_data
            A_consistency = sparse.diags(psi_consistency.flatten(),shape=(h*w,h*w))
            b_consistency = A_consistency.dot((A_l-A_reference).flatten())
            
            
            USE_WEIGHTED = True
            ###################################################
            #
            # 1st order term
            #
            ###################################################
            if USE_WEIGHTED:
                # Weighted case
                grad_A_1storder = M_GRAD.dot(A_l.flatten())
                grad_A_1storder_reshaped = grad_A_1storder.reshape((-1,h*w))
                weightarray_1storder = np.c_[weights_x.flatten(), weights_y.flatten()].T

                psi_smooth_1storder = robust_deriv((weightarray_1storder * grad_A_1storder_reshaped**2).sum(axis=0))
                
                P = sparse.diags(
                        np.r_[ (weights_x.flatten()*psi_smooth_1storder).flatten(),
                               (weights_y.flatten()*psi_smooth_1storder).flatten()])
                A_smooth_1storder = -M_DIV.dot(P).dot(M_GRAD)

            else:                
                # Unweighted case
                grad_A_1storder = M_GRAD.dot(A_l.flatten())
                grad_A_1storder_reshaped = grad_A_1storder.reshape((-1,h*w))
                psi_smooth_1storder = robust_deriv((grad_A_1storder_reshaped**2).sum(axis=0))
                P = sparse.diags(np.r_[psi_smooth_1storder.flatten(),psi_smooth_1storder.flatten()])
                A_smooth_1storder = -M_DIV.dot(P).dot(M_GRAD)

            b_smooth_1storder = A_smooth_1storder * A.flatten()


            ###################################################
            #
            # 2nd order term
            #
            ###################################################
            if USE_WEIGHTED:
                # Weighted case
                
                # Somewhat hacky.
                weights_xy = np.minimum(weights_x,weights_y)

                # This contains the 2nd derivative images, vectorized and stacked
                grad_A_2ndorder = M_GRAD_2NDORDER.dot(A_l.flatten())

#                weightarray_2ndorder = np.c_[weights_x.flatten(),weights_xy.flatten(),weights_y.flatten()].T
                weightarray_2ndorder = np.c_[weights_x.flatten(),weights_y.flatten()].T
                                             
                                             
                # The components are summed within the robust function, so we first
                # need to reshape grad_A_2ndorder, so that it contains the vectorized
                # derivative images as rows.
                grad_A_2ndorder_reshaped = grad_A_2ndorder.reshape((-1,h*w))
                psi_smooth_2ndorder = robust_deriv( (weightarray_2ndorder * grad_A_2ndorder_reshaped**2).sum(axis=0) )
                P = sparse.diags(np.r_[
                    psi_smooth_2ndorder.flatten() * weights_x.flatten(),
                    #psi_smooth_2ndorder.flatten() * weights_xy.flatten(),
                    psi_smooth_2ndorder.flatten() * weights_y.flatten()])
                
                A_smooth_2ndorder = -M_DIV_2NDORDER.dot(P).dot(M_GRAD_2NDORDER)
            else:
                # Unweighted case

                # This contains the 2nd derivative images, vectorized and stacked
                grad_A_2ndorder = M_GRAD_2NDORDER.dot(A_l.flatten())

                # The components are summed within the robust function, so we first
                # need to reshape grad_A_2ndorder, so that it contains the vectorized
                # derivative images as rows.
                grad_A_2ndorder_reshaped = grad_A_2ndorder.reshape((-1,h*w))
                psi_smooth_2ndorder = robust_deriv( (grad_A_2ndorder_reshaped**2).sum(axis=0) )
                P = sparse.diags(np.r_[
                    psi_smooth_2ndorder.flatten(),
                    #psi_smooth_2ndorder.flatten(),
                    psi_smooth_2ndorder.flatten()])
                
                A_smooth_2ndorder = -M_DIV_2NDORDER.dot(P).dot(M_GRAD_2NDORDER)

            b_smooth_2ndorder = A_smooth_2ndorder * A.flatten()

            #
            #
            # Compute energy
            #
            #
            # Compute energy
            
            E_data = robust_ev(Iz_bar**2).sum()
            E_consistency = robust_ev((A_l-A_reference)**2).sum()
            E_1storder = robust_ev((grad_A_1storder_reshaped**2).sum(axis=0)).sum()
            E_2ndorder = robust_ev((grad_A_2ndorder_reshaped**2).sum(axis=0)).sum()

            
            energy = E_data + lambda_1storder * E_1storder + lambda_2ndorder * E_2ndorder + lambda_consistency * E_consistency
            n_inframe = (in_frame==1).sum()
            
            pdb.set_trace()
            
            print('({0:02d}) EpPx={1:2.5f} -- Data={2:2.1f} -- Consistency={3:2.1f} -- 1st ord={4:2.1f} -- 2nd ord={5:2.1f}'.format(
                  k,energy/(h*w),
                  E_data,
                  lambda_consistency*E_consistency,
                  lambda_1storder * E_1storder,
                  lambda_2ndorder * E_2ndorder))
            energies.append(energy/(h*w))
                
            
            
            #pdb.set_trace()
            
            # Solve system -> obtain da
            
            A_combined = A_data + lambda_1storder * A_smooth_1storder + lambda_2ndorder * A_smooth_2ndorder + lambda_consistency * A_consistency
            b_combined = -b_data - lambda_1storder * b_smooth_1storder - lambda_2ndorder * b_smooth_2ndorder - lambda_consistency * b_consistency
            
            da = spsolve(A_combined, b_combined,use_umfpack=False)
            
            # Clipping of da -- this is a bit questionable.
            da = np.clip(da,-1,1)
    
            da = da.reshape(A.shape)        
            
            # Median filtering of da
            da = ndimage.median_filter(A + da, size=5) - A
            
            
        #print('Updating...')
        A = A + da
        
        
    #
    # Debugging
    #
    if False:
        I2w_init, mask_init = evaluate_data_term(A_init, I1, I2, H, B, q, mu)
        I2w_opt, mask_opt = evaluate_data_term(A, I1, I2, H, B, q, mu)
        A_new = A
        pdb.set_trace()
        
    return A, energies
