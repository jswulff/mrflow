#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

import cv2

from chumpy import ch
from scipy import optimize

sys.path.append(os.environ['MRFLOW_HOME'])

# Local imports
from utils import flow_homography as fh

from utils import plot_debug as pd
from utils import print_exception as pe
from utils.spyder_debug import *
from utils import compute_figure

from rigidity import inference
from initialization import refine_A

def parallax2normedstructure(norm_parallax, q, B, mu):
    """
    Convert parallax (given in pixels of motion towards epipole) to structure
    """
    h,w = norm_parallax.shape
    y,x = np.mgrid[:h,:w]
    dst = np.sqrt((x-q[0])**2 + (y-q[1])**2)
    An = (norm_parallax) / (B*(norm_parallax/mu - dst / mu))
    return An



def compute_structure(images,flow,rigidity,occlusions_precomputed,homographies,epipoles,params):
    """
    Build cost volumes for both separate frames, where the third dimension
    of the cost volume is the structure, *not* the displacement.
    """

    if not params.debug_use_rigidity_gt:
        rigidity_thresholded = cv2.erode((rigidity>0.5).astype('uint8'),
                                         np.ones((3,3),np.uint8))>0
    else:
        rigidity_thresholded = rigidity

    h,w = images[0].shape[:2]
    y,x = np.mgrid[:h,:w]

    I1 = images[1]

    images_neighbors = [images[0],images[2]]

    n_frames = 2

    residual_flow_array = []
    epipoles_new_array = []
    parallax_array = []
    mu_array = [] # Holding the average of distances
    dists_normed_array = [] # Holding the distances, divided by the average

    #
    # First pass: Determine new epipoles, forward and backwards corrected flow, parallax, and normalization variables
    #
    for f in range(n_frames):
        u,v = flow[f]
        q = epipoles[f]
        H = homographies[f]
        I2 = images_neighbors[f]

        # Compute residual flow after correction by homography
        u_res, v_res = fh.get_residual_flow(x,y,u,v,H)

        # New refined epipole
        q_new = correct_epipole(u_res,v_res,q,rigidity_thresholded>0)
        print('Q old: {}'.format(q))
        print('Q new: {}'.format(q_new))

        # Get parallax from residual flow
        parallax, foe_vectors, foe_dists = flow2parallax(u_res,v_res,q_new)

        mu = foe_dists.mean()

        # Save computation results
        residual_flow_array.append([u_res,v_res])
        epipoles_new_array.append(q_new)
        parallax_array.append(parallax)
        mu_array.append(mu)
        dists_normed_array.append(foe_dists/mu)


        if params.debug_save_frames:
            pd.plot_flow(u,v,'[04 Structure Computation] [00] Initial flow frame {0:01d}'.format(f),outpath=params.tempdir)
            pd.plot_flow(u_res,v_res,'[04 Structure Computation] [00] Residual flow frame {0:01d}'.format(f),outpath=params.tempdir)
            pd.plot_image(parallax, '[04 Structure Computation] [00] Parallax frame {0:01d}'.format(f),colorbar=True,vmin=-5,vmax=5,outpath=params.tempdir)
            pd.plot_image(foe_dists/mu, '[04 Structure Computation] [00] Correction image (foe_dists over mu) frame {0:01d}'.format(f),colorbar=True,vmin=0.0,vmax=2,outpath=params.tempdir)

    #
    # Compute rigidity unaries
    #

    # The simple CNN rigidity
    p_rigidity_cnn = rigidity.copy()

    # Rigidity based on the direction of motion
    sigma_direction = params.rigidity_sigma_direction

    # Check pixels for which the motion becomes very small - they are
    # likely to lie on the plane, and thus likely to be rigid.
    small_bwd = np.sqrt(residual_flow_array[0][0]**2 + residual_flow_array[0][1]**2) < 0.1
    small_fwd = np.sqrt(residual_flow_array[1][0]**2 + residual_flow_array[1][1]**2) < 0.1
    
    p_rigidity_direction_bwd = inference.get_unaries_rigid(
        x,y,
        residual_flow_array[0][0],residual_flow_array[0][1], # u,v
        epipoles_new_array[0][0], epipoles_new_array[0][1], #qx,qy
        sigma=sigma_direction,
        )
    p_rigidity_direction_fwd = inference.get_unaries_rigid(
        x,y,
        residual_flow_array[1][0],residual_flow_array[1][1], # u,v
        epipoles_new_array[1][0], epipoles_new_array[1][1], #qx,qy
        sigma=sigma_direction,
        )

    p_rigidity_direction_bwd[small_bwd] = 0.6
    p_rigidity_direction_fwd[small_fwd] = 0.6

    occ_bwd = occlusions_precomputed[0]==1
    occ_fwd = occlusions_precomputed[1]==1

    # Remove occlusions from fwd,bwd
    p_rigidity_direction_fwd[occ_fwd] = 0.5
    p_rigidity_direction_bwd[occ_bwd] = 0.5


    # Merge, based on occlusions
    p_rigidity_direction = (p_rigidity_direction_bwd + p_rigidity_direction_fwd) / 2.0
    # Exclude bwd occlusions
    p_rigidity_direction[occ_bwd] = 0.25 + 0.5 * p_rigidity_direction_fwd[occ_bwd]
    # Exclude fwd occlusions
    p_rigidity_direction[occ_fwd] = 0.25 + 0.5 * p_rigidity_direction_bwd[occ_fwd]
    # Where both are occluded, simply use non-informative prior.
    p_rigidity_direction[occ_bwd*occ_fwd] = 0.5

    # Threshold rigidity. This is just to temporally exclude some rigid
    # parts for the structure refinement / computation that is used to
    # initialize the actual optimization.

    weight_cnn = params.rigidity_weight_cnn
    p_rigidity_thresholded = (weight_cnn * p_rigidity_cnn + (1-weight_cnn) * p_rigidity_direction) > 0.5

    if params.debug_save_frames:
        pd.plot_image(p_rigidity_cnn, '[04 Structure Computation] [01] Rigidity: CNN', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')
        pd.plot_image(p_rigidity_direction_fwd, '[04 Structure Computation] [01] Rigidity: Direction, fwd', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')
        pd.plot_image(p_rigidity_direction_bwd, '[04 Structure Computation] [01] Rigidity: Direction, bwd', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')
        pd.plot_image(p_rigidity_direction, '[04 Structure Computation] [01] Rigidity: Direction', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')
        pd.plot_image((p_rigidity_direction + p_rigidity_cnn)/2.0, '[04 Structure Computation] [01] Rigidity: Combined', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')
        pd.plot_image(p_rigidity_thresholded, '[04 Structure Computation] [01] Rigidity: Thresholded', outpath=params.tempdir,vmin=0,vmax=1,cmap='bwr')


    if p_rigidity_thresholded.sum() < p_rigidity_thresholded.size * 0.25:
        raise Exception('TooMuchNonRigid')


    #
    # Compute backward B
    #
    # Compute valid pixels
    pv = np.logical_and(
            np.logical_and(
                np.abs(parallax_array[0])>0.1, np.abs(parallax_array[1])>0.1),
            p_rigidity_thresholded>0)

    mu0 = mu_array[0]
    mu1 = mu_array[1]

    #
    # Compute b_fwd so that the normed structure in forward direction is scaled properly.
    #
    if pv.sum() < 100:
        B_fwd = 1.0
    else:
        mask = pv
        structure_normed = parallax2normedstructure(parallax_array[1], epipoles_new_array[1], 1.0, mu_array[1])
        if params.scale_structure == 1:
            # Use MAD for scaling
            A_std_robust = 1.426 * np.median(np.abs(structure_normed[mask] - np.median(structure_normed[mask])))
            B_fwd = A_std_robust
        elif params.scale_structure == 2:
            # Use STD for scaling
            A_std_robust = structure_normed[mask].std()
            B_fwd = A_std_robust
        else:
            # Use previous default.
            B_fwd = 1.0


    # Compute initial B_bwd so that the resulting A_bwd and A_fwd match well.

    B_all_array = None
    if pv.sum() < 100:
        # Default value if not enough points are off the H
        B_b = -B_fwd
        print('(WW) Most points on H, occluded, or non-rigid. Using default value for B_b.')
    else:
        par0 = parallax_array[0][pv]
        par1 = parallax_array[1][pv]
        dist_norm0 = dists_normed_array[0][pv]
        dist_norm1 = dists_normed_array[1][pv]

        # A_fwd as the target
        target = par1 / (B_fwd * (par1/mu1 - dist_norm1))

        # A_bwd is the template that we want to match
        template = mu1/mu0 * par0 / ( (par0/mu0 - dist_norm0))

        B_all_array = template/target
        B_b = np.median(B_all_array)

    B_array = [B_b, B_fwd]


    if params.debug_save_frames:
        B_image = np.zeros_like(parallax_array[0])
        if B_all_array is not None:
            B_image[pv] = B_all_array
            pd.plot_image(B_image, '[04 Structure Computation] [02] Array of B_bwd', colorbar=True,vmin=-2,vmax=0,outpath=params.tempdir)
        pd.plot_image(pv, '[04 Structure Computation] [02] Valid pixels', colorbar=True,vmin=0,vmax=1,outpath=params.tempdir)


    #
    # Compute normed structure
    #
    structure_array = []
    structure_center_removed_array = []

    for f in range(n_frames):
        q = epipoles_new_array[f]
        b = B_array[f]
        structure_normed = parallax2normedstructure(parallax_array[f], q, b, mu_array[f])

        # Save output
        structure_array.append(structure_normed)

        if params.debug_save_frames:
            if f==0:
                S = structure_normed * mu1/mu0
            else:
                S = structure_normed
            pd.plot_image(S, '[04 Structure Computation] [03] Normed structure in frame {}'.format(f), colorbar=True, vmin=-5,vmax=5,outpath=params.tempdir)


    #
    # Refinement of backward structure (structure_array[0])
    #
    if params.nonlinear_structure_initialization > 0:
        print('========== Refining backwards motion and structure ==========')
        A_bwd_new, H_bwd_new, B_bwd_new, q_bwd_new = refine_A.refine_A(
            flow[0][0], flow[0][1], # Bwd flow
            homographies[0],
            B_array[0],
            epipoles_new_array[0],
            mu_array[0],
            mu_array[1], # Reference mu
            structure_array[1],
            p_rigidity_thresholded>0,
            (occlusions_precomputed[0]==0)*(occlusions_precomputed[1]==0),
        )

        #
        # For now, we do not refine the parameter H, B, q in forward direction,
        # since this usually results in a worse EPE performance.
        #
        if False:
            print('========== Refining forward motion and structure ==========')
            A_fwd_new, H_fwd_new, B_fwd_new, q_fwd_new = refine_A.refine_A(
                flow[1][0], flow[1][1], # Fwd flow
                homographies[1],
                B_array[1],
                epipoles_new_array[1],
                mu_array[1],
                mu_array[0], # Reference mu
                A_bwd_new,
                p_rigidity_thresholded>0,
                (occlusions_precomputed[0]==0)*(occlusions_precomputed[1]==0),
                refine_B=False,
            )

        else:
            A_fwd_new = structure_array[1]
            H_fwd_new = homographies[1]
            B_fwd_new = B_array[1]
            q_fwd_new = epipoles_new_array[1]

        print('====== Refinement results ======')
        print('Backward:')
        print('B:\t{}\t=>\t{}'.format(B_array[0], B_bwd_new))
        print('H:')
        print(homographies[0])
        print('-')
        print(H_bwd_new)
        print('')
        print('Forward:')
        print('B:\t{}\t=>\t{}'.format(B_array[1], B_fwd_new))
        print('H:')
        print(homographies[1])
        print('-')
        print(H_fwd_new)
        print('')

        # Show some debugging
        structure_difference_before = np.abs(structure_array[0] * mu1/mu0 - structure_array[1])
        structure_difference_after_bwd = np.abs(A_bwd_new * mu1/mu0 - structure_array[1])
        structure_difference_after_fwd = np.abs(A_bwd_new * mu1/mu0 - A_fwd_new)
        print('')
        print('(MM) Mean/Median of structure difference before optimization: {}\t{}'.format(
            structure_difference_before.mean(), np.median(structure_difference_before)))
        print('(MM) Mean/Median of structure difference after backward optimization: {}\t{}'.format(
            structure_difference_after_bwd.mean(), np.median(structure_difference_after_bwd)))
        print('(MM) Mean/Median of structure difference after forward optimization: {}\t{}'.format(
            structure_difference_after_fwd.mean(), np.median(structure_difference_after_fwd)))

        # Save output
        if params.debug_save_frames:
            S = A_bwd_new * mu_array[1]/mu_array[0]
            S_fwd = A_fwd_new
            pd.plot_image(S, '[04 Structure Computation] [04] Refined structure in frame 0', colorbar=True, vmin=-5,vmax=5,outpath=params.tempdir)
            pd.plot_image(A_fwd_new, '[04 Structure Computation] [04] Refined structure in frame 1', colorbar=True, vmin=-5,vmax=5,outpath=params.tempdir)

            I0_warped = cv2.warpPerspective(images[0].astype('float32'),
                                            H_bwd_new,
                                            (w,h),
                                            flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            I2_warped = cv2.warpPerspective(images[2].astype('float32'),
                                            H_fwd_new,
                                            (w,h),
                                            flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            pd.plot_image(I0_warped, '[04 Structure Computation] [04] Warped (refined) frame 0', outpath=params.tempdir)
            pd.plot_image(images[1], '[04 Structure Computation] [04] Warped (refined) frame 1', outpath=params.tempdir)
            pd.plot_image(I2_warped, '[04 Structure Computation] [04] Warped (refined) frame 2', outpath=params.tempdir)
            pd.plot_image(structure_difference_before, '[04 Structure Computation] [04] Structure difference before opt', outpath=params.tempdir, vmin=0, vmax=5)
            pd.plot_image(structure_difference_after_bwd, '[04 Structure Computation] [04] Structure difference after opt', outpath=params.tempdir, vmin=0, vmax=5)


        structure_array[0] = A_bwd_new
        homographies[0] = H_bwd_new
        B_array[0] = B_bwd_new
        epipoles_new_array[0] = q_bwd_new

        structure_array[1] = A_fwd_new
        homographies[1] = H_fwd_new
        B_array[1] = B_fwd_new
        epipoles_new_array[1] = q_fwd_new



    #
    # Compute occlusions
    #

    if params.occlusion_reasoning > 0:
        occ = occlusions_precomputed
    else:
        occ = [np.ones((h,w))>0, np.ones((h,w))>0]


    # 
    # With the computed refined backward structure, refine rigidity map
    # using CNN + direction + structure difference.
    #
    if params.rigidity_use_structure:

        print('========== Refining rigidity map ==========')
        sigma_structure = params.rigidity_sigma_structure

        A_diff = structure_array[0] * mu1/mu0 - structure_array[1]
        p_rigidity_structure = np.exp(-(A_diff/sigma_structure)**2)

        structure_invalid = ((occ[0]==0) * (occ[1]==0))==0
        p_rigidity_structure[structure_invalid] = 0.5

        p_rigidity_motion = p_rigidity_direction * p_rigidity_structure
        p_rigidity_motion[structure_invalid] = 0.25 + 0.5 * p_rigidity_direction[structure_invalid]

        weight_cnn = params.rigidity_weight_cnn
        p_rigidity = weight_cnn * p_rigidity_cnn + (1-weight_cnn) * p_rigidity_motion

        # Construct global rigidity unaries from motion and CNN, and use
        # TRWS to do the inference step.

        unaries = np.dstack((1-p_rigidity, p_rigidity))

        LAMBD=1.1
        rigidity_refined = inference.infer_mrf(images[1], unaries, lambd=LAMBD)
        rigidity_refined = rigidity_refined>0.5

        if params.debug_compute_figure == 3:
            compute_figure.plot_figure_3(images[1], rigidity, p_rigidity_direction, p_rigidity_structure, rigidity_refined)
            sys.exit(1)

        # Plot difference
        if params.debug_save_frames:
            pd.plot_image(np.abs(A_diff), '[04 Structure Computation] [05] Structure difference', colorbar=True, vmin=0,vmax=5,outpath=params.tempdir)
            pd.plot_image(p_rigidity_structure,
                        '[04 Structure Computation] [05] Rigidity: Structure',
                    vmin=0,vmax=1,cmap='bwr',outpath=params.tempdir)
            pd.plot_image(p_rigidity_motion,
                        '[04 Structure Computation] [05] Rigidity: Combined motion',
                    vmin=0,vmax=1,cmap='bwr',outpath=params.tempdir)
            pd.plot_image(p_rigidity,
                        '[04 Structure Computation] [05] Rigidity: Combined probability',
                    vmin=0,vmax=1,cmap='bwr',outpath=params.tempdir)
            pd.plot_image(rigidity_refined,
                          '[04 Structure Computation] [05] Rigidity: Final estimate',
                          vmin=0,vmax=1,cmap='bwr',outpath=params.tempdir)




    # As above: If we want to use the GT rigidity, dont refine
    if params.debug_use_rigidity_gt:
        rigidity_refined = rigidity

    if (rigidity_refined==1).sum() < 0.25 * rigidity_refined.size:
        # This was probably a bad alignment - the structure does not match
        raise Exception('TooFewStructureMatches')


    if params.debug_compute_figure == 91:
        compute_figure.plot_figure_factorization_a(homographies, flow)

    if params.debug_compute_figure == 93:
        compute_figure.plot_figure_video_rigidity_example(images[1], rigidity_refined)
        sys.exit(1)





    return [structure_array,
            homographies,
            mu_array,
            B_array,
            epipoles_new_array,
            rigidity_refined,
            occ]



def correct_epipole(u,v,q,rigidity):
    """
    Given the aligned flow fields and the epipole, update the epipole so that
    the normalized parallax does not cross the epipole.

    Parameters
    ----------
    u,v     : aligned flow fields
    q       : initial epipole

    Returns
    -------
    q_new   : Updated epipole

    """
    h,w = u.shape

    xmin = int(q[0]) - 10
    xmax = int(q[0]) + 10
    ymin = int(q[1]) - 10
    ymax = int(q[1]) + 10

    # Check if Q is inside image
    if xmin < 0 or xmax >= w or ymin < 0 or ymax >= h:
        return q

    y,x = np.mgrid[ymin:ymax+1,xmin:xmax+1]
    u = u[ymin:ymax+1,xmin:xmax+1]
    v = v[ymin:ymax+1,xmin:xmax+1]
    r_around_q = rigidity[ymin:ymax+1,xmin:xmax+1]

    u = u[r_around_q>0]
    v = v[r_around_q>0]
    x = x[r_around_q>0]
    y = y[r_around_q>0]

    if (r_around_q==0).sum() > (r_around_q.size/2):
        print('(WW) Too many non-rigid pixels close to epipole')
        return q

    # Method to be optimized
    def fun(q_estimated):
        q_ch = ch.array(q_estimated)
        u_f = q_ch[0] - x
        v_f = q_ch[1] - y
        dists_squared = ch.maximum(1e-3,u_f**2 + v_f**2)
        parallax_unnormed = (u*u_f + v*v_f)
        #df = ch.maximum(0, parallax_unnormed - dists_squared)**2
        df = ch.abs(parallax_unnormed)/dists_squared
        err = df.sum()
        derr = err.dr_wrt(q_ch).copy().flatten()
        return err(),derr

    x0 = q
    res = optimize.minimize(fun,x0=x0,jac=True,method='BFGS',options={'disp': False})
    print(res)
    q_new = res.x if np.linalg.norm(res.x - q) < 10 else q

    return q_new





def flow2parallax(u,v,q):
    """
    Given the flow fields (after correction!) and the epipole,
    return:
    - The normalized parallax (HxW array)
    - The vectors pointing to the epipoles (HxWx2 array)
    - The distances of all points to the epipole (HxW array)
    """
    h,w = u.shape
    y,x = np.mgrid[:h,:w]

    u_f = q[0] - x
    v_f = q[1] - y

    dists = np.sqrt(u_f**2 + v_f**2)
    u_f_n = u_f / np.maximum(dists,1e-3)
    v_f_n = v_f / np.maximum(dists,1e-3)

    parallax = u * u_f_n + v * v_f_n

    return parallax, np.dstack((u_f_n, v_f_n)), dists










