#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

import cv2

sys.path.append(os.environ['MRFLOW_HOME'])

# Local imports
from initialization import homographies as init_homographies
from utils import plot_debug as pd

def align_frames(images,matches_all_frames,matches_pairwise, params):
    """ Given images and set of matches, compute best alignment homographies.
    """
    h,w = images[0].shape[:2]

    if params.alignment_ransac:
        #
        # This is the old fitting.
        #
        homographies_init = init_homographies.compute_initial_homographies(matches_all_frames, ransacReprojThreshold=params.alignment_ransac_threshold)

    else:
        #
        # This one first estimates F, then selects three points that best
        # span a plane, and constructs H from there as described in Z&H.
        #
        homographies_init = init_homographies.compute_initial_homographies_via_F_explicit(matches_all_frames, ransacReprojThreshold=params.alignment_ransac_threshold)


    # Check if we want to add the additional homography refinement.
    # This tries to make sure that the homographies correspond to real planes,
    # i.e. that all parallax points to the same epipole.
    if params.refine_coplanarity > 0:
        homographies_refined = init_homographies.refine_homographies(matches_all_frames, homographies_init,(h,w),error_type=params.alignment_error_type)
    else:
        homographies_refined = homographies_init

    # Compute final estimates for foci of expansion
    epipoles = init_homographies.compute_foes_svd_irls(matches_all_frames, homographies_refined)


    # Detect which features are located on the refined homographies
    plane_global = init_homographies.compute_inliers(matches_all_frames, homographies_refined)
    plane_local = inliers_global_to_pairwise(plane_global, matches_all_frames, matches_pairwise)

    homographies = homographies_refined

    for i,h_ in enumerate(homographies_init):
        print('-- Initial homography for frame {} --'.format(i))
        print(h_)

    print('--- Epipoles after refinement: ---')
    print(epipoles[0])
    print(epipoles[1])

    for i,h_ in enumerate(homographies):
        print('-- Refined homography for frame {} --'.format(i))
        print(h_)

    #
    # Image output
    #
    if params.debug_save_frames:
        n_frames = len(images)-1
        frames = [0,2]
        for f in frames:
            if f > 1:
                H = homographies[f-1]
            else:
                H = homographies[f]

            Iw = cv2.warpPerspective(images[f].astype('float32'),
                    H,
                    (w,h),
                    flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            pd.plot_image(Iw, '[03 Alignment] Warped (refined) frame {}'.format(f), outpath=params.tempdir)

            # Warp with initial
            H = homographies_init[f-1] if f > 1 else homographies_init[f]
            Iw = cv2.warpPerspective(images[f].astype('float32'),
                   H,
                   (w,h),
                   flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            pd.plot_image(Iw, '[03 Alignment] Warped (initial) frame {}'.format(f), outpath=params.tempdir)

           

        pd.plot_image(images[1].astype('float32'), '[03 Alignment] Warped (initial) frame 1', outpath=params.tempdir)
        pd.plot_image(images[1].astype('float32'), '[03 Alignment] Warped (refined) frame 1', outpath=params.tempdir)

        # Generate point image.
        masks = [plane_global==0, plane_global==1]
        pd.plot_scatter(matches_all_frames[:,:,0], '[03 Alignment] Feature matches (refined)', masks=masks, I=images[1],outpath=params.tempdir)

        # Generate features using initial homography
        plane_global_init = init_homographies.compute_inliers(matches_all_frames, homographies_init)
        masks = [plane_global_init==0,plane_global_init==1]
        pd.plot_scatter(matches_all_frames[:,:,0], '[03 Alignment] Feature matches (initial)', masks=masks, I=images[1],outpath=params.tempdir)

    #
    # End image output code.
    #


    return homographies, epipoles, plane_global


def inliers_global_to_pairwise(inliers_global, matches_global, matches_pairwise):
    """
    Extract pairwise inlier masks.

    Parameters
    ----------
    inliers_global : binary array
        Global inlier mask
    matches_global : array_like, size (N_FEATURES x 2 x FRAMES)
        Global matches. matches_global[:,:,0] are the features in the
        reference frame.
    matches_pairwise : list of array_like, each of size (N_FEATURES x 2 x 2)
        Pairwise features.

    Returns
    -------
    inliers_local : list of array_like, each of size N_FEATURES

    """
    coords_inliers_first_frame = matches_global[inliers_global,:,0]
    list_inliers_first_frame = [tuple(c) for c in coords_inliers_first_frame]
    n_i = len(matches_pairwise)
    inliers_pairwise = []
    for m_p in matches_pairwise:
        coords_current_pair = m_p[:,:,0]
        inliers_cur = np.zeros(m_p.shape[0])
        for i,p in enumerate(coords_current_pair):
            if tuple(p) in list_inliers_first_frame:
                inliers_cur[i] = 1
        inliers_pairwise.append(np.array(inliers_cur))
    return inliers_pairwise










