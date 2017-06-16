#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

import cv2


sys.path.append(os.environ['MRFLOW_HOME'])
# Local imports
from utils import plot_debug as pd
from utils import print_exception as pe
from utils.spyder_debug import *
from utils import compute_figure

from structure_refinement import optimize_structure_multi_scale


def optimize(images, structures, rigidity, epipoles, homographies, B, mu, occlusions, params):

    occlusions_bwd, occlusions_fwd = occlusions
    occlusions_bwd = cv2.medianBlur(occlusions_bwd.astype('uint8'), ksize=3)>0
    occlusions_fwd = cv2.medianBlur(occlusions_fwd.astype('uint8'), ksize=3)>0
    occlusions_both = occlusions_bwd*occlusions_fwd

    # Kind of a hack -- just used to initialize the optimized structure.
    occlusions_fwd_only = (occlusions_bwd==0)*(occlusions_fwd>0)
    occlusions_bwd_only = (occlusions_fwd==0)*(occlusions_bwd>0)
    no_occlusion = (occlusions_fwd_only==0)*(occlusions_bwd_only==0)

    if params.occlusion_reasoning == 0:
        # In this case, we cannot simply blend the structures together, since this would destroy
        # the end estimate due to wrong backward estimation.
        # Instead, just use the forward structure.
        occlusions_bwd_only[:] = True
        no_occlusion[:] = False

    # Merge structures
    S0 = structures[0] * mu[1]/mu[0]
    S1 = structures[1]
    structure_optimized = occlusions_fwd_only * S0 + occlusions_bwd_only * S1 + no_occlusion * (S0+S1)/2.0


    #
    # Figure for video
    #
    if params.debug_compute_figure == 91:
        # Factorization figure
        compute_figure.plot_figure_factorization_b(
            images,
            structures,
            structure_optimized,
            rigidity)
        sys.exit(1)


    #
    # Print some statistics on init structure
    #
    mad = lambda x : 1.426 * np.median(np.abs(x - np.median(x)))
    A_std = structure_optimized.std()
    A_mad = mad(structure_optimized)
    structmax = np.percentile(structure_optimized,95)
    structmin = np.percentile(structure_optimized,5)
    print('(MM) Statistics of input structure: 5th perc={0:2.3f}\t95th perc={1:2.3f}\tstd={2:2.3f}\tmad={3:2.3f}'.format(structmin,structmax,A_std,A_mad))


    #
    # False to override the optimization to speed up things.
    #
    if params.override_optimization:
        return [structure_optimized, ]
    

    #
    # Now we have the initial structure estimation in structure_optimized,
    # as well as the occlusion map estimates.
    #
    A_optimized, energies = optimize_structure_multi_scale.optimizeStructureMultiScale(
            structure_optimized, # =A_init
            S0, S1, # = A_bwd, A_fwd
            occlusions_bwd, occlusions_fwd,
            rigidity, 
            images,
            homographies, B, epipoles, mu,
            params)

    if params.debug_save_frames:
        pd.plot_image(S0, '[05 Optimization Variational] Backwards structure', vmin=structmin,vmax=structmax,outpath=params.tempdir)
        pd.plot_image(S1, '[05 Optimization Variational] Forwards structure', vmin=structmin,vmax=structmax,outpath=params.tempdir)
        pd.plot_image(structure_optimized, '[05 Optimization Variational] Optimized structure (init)', vmin=structmin,vmax=structmax,outpath=params.tempdir)
        pd.plot_image(occlusions_bwd.astype('int') + occlusions_fwd.astype('int')*2,
                '[05 Optimization Variational] Source of structure',outpath=params.tempdir)
        pd.plot_image(A_optimized, '[05 Optimization Variational] Optimized structure', vmin=structmin, vmax=structmax, outpath=params.tempdir)

    if params.debug_compute_figure == 92:
        compute_figure.plot_figure_video_structure(structures, structure_optimized, A_optimized, rigidity)


    return [A_optimized,]

