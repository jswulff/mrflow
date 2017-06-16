#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:15:30 2016

@author: jonas
"""

import os,sys,pdb
import warnings

import numpy as np
import cv2

sys.path.append(os.environ['MRFLOW_HOME'])

#
# Project-local imports
#
from utils.spyder_debug import *
from utils import print_exception as pe

#
# Module-local imports
#
import optimize_structure_single_scale

def mad(x):
    return np.median(np.abs(x-np.median(x)))

def optimizeStructureMultiScale(A_init,
        A_bwd,
        A_fwd,
        occlusions_bwd,
        occlusions_fwd,
        rigidity,
        images, #array
        H_ar, #array
        B_ar, #array
        q_ar, #array
        mu_ar, #array
        params):

    lambda_1storder = params.variational_lambda_1storder
    lambda_2ndorder = params.variational_lambda_2ndorder
    lambda_consistency = params.variational_lambda_consistency
    N_outer = params.variational_N_outer
    N_inner = params.variational_N_inner
    scaleFactor = params.variational_scale_factor
    N_scales = params.variational_N_scales
    last_scale = params.variational_last_scale


    scales = scaleFactor ** np.arange(last_scale, N_scales)
    
    energies = []

    A_scale_factor = 1.0

    # downscaling: use INTER_AREA
    A_scaled = cv2.resize(A_init, dsize=None, fx=scales[-1],fy=scales[-1], interpolation=cv2.INTER_AREA)

    for s in reversed(scales):

        # Scale down images, fwd/bwd A (for consistency), estimated A, occlusion maps.
        images_scaled = [cv2.resize(I, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for I in images]

        A_fwd_scaled = cv2.resize(A_fwd, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        A_bwd_scaled = cv2.resize(A_bwd, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        # Upscaling: Use cubic resampling
        A_scaled = cv2.resize(A_scaled, dsize=(A_fwd_scaled.shape[1],A_fwd_scaled.shape[0]),interpolation=cv2.INTER_CUBIC)

        # For the occlusion maps, we use bilinear interpolation, and remove all pixels > 0.
        # This somewhat dilates the occlusion maps, but better to err in this direction.
        occlusions_fwd_scaled = cv2.resize(occlusions_fwd.astype('float32'), dsize=None, fx=s, fy=s) > 0
        occlusions_bwd_scaled = cv2.resize(occlusions_bwd.astype('float32'), dsize=None, fx=s, fy=s) > 0

        # Since by this we might have introduced spurious "both-occluded" pixels,
        # filter those out.

        both_occluded = occlusions_fwd_scaled * occlusions_bwd_scaled
        occlusions_fwd_scaled[both_occluded] = 0
        occlusions_bwd_scaled[both_occluded] = 0
        occlusions_both = both_occluded

        # Scale down rigidity.
        # Since the rigidity map is 1 where things are rigid, a slight erosion is necessary
        # after downscaling (similar effect to above).
        rigidity_scaled = cv2.resize(rigidity.astype('float32'), dsize=None, fx=s, fy=s)==1

        print('*** Running scale {}, shape {} ***'.format(s, A_scaled.shape[:2]))
        
       
        #
        # Scale down motion data
        #
        B_ar_scaled = [B / s for B in B_ar]
        mu_ar_scaled = [mu * s for mu in mu_ar]
        q_ar_scaled = [q * s for q in q_ar]
        
        # Scale down homographies
        scalingMatrixH = np.array([[1.0, 1.0, s],
                                   [1.0, 1.0, s],
                                   [1.0/s, 1.0/s, 1.0]])
        H_ar_scaled = [H * scalingMatrixH for H in H_ar]
        
        with warnings.catch_warnings():
            #warnings.simplefilter("error")
            try:
                A_scaled, energies_cur = optimize_structure_single_scale.optimizeStructureSingleScale(
                    A_scaled, # Initial A estimate
                    A_bwd_scaled,
                    A_fwd_scaled,
                    occlusions_bwd_scaled,
                    occlusions_fwd_scaled,
                    occlusions_both,
                    rigidity_scaled,
                    images_scaled,
                    H_ar_scaled,
                    B_ar_scaled,
                    q_ar_scaled,
                    mu_ar_scaled,
                    lambda_1storder,
                    lambda_2ndorder,
                    lambda_consistency,
                    A_scale_factor=A_scale_factor,
                    N_outer=N_outer,
                    N_inner=N_inner,
                    params=params)

                energies += energies_cur
            except:
                sys.stderr.write('\n\n (EE) Warning in {} \n\n'.format(params.tempdir))
                sys.stderr.write('Params:\n')
                sys.stderr.write(repr(params))
                sys.stderr.flush()
                pe.print_exception()

        
    if not (A_scaled.shape[0]==A_init.shape[0] and A_scaled.shape[1]==A_init.shape[1]):
        A_scaled = cv2.resize(A_scaled, dsize=(A_init.shape[1],A_init.shape[0]),interpolation=cv2.INTER_LANCZOS4)
    
    return A_scaled,energies
