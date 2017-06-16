#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

import cv2
from skimage import img_as_float
from skimage import morphology

from sklearn.cluster import estimate_bandwidth, MeanShift

from chumpy import ch
from scipy import optimize


cpath = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(cpath)

# Local imports
from utils import flow_homography as fh

from utils import plot_debug as pd
from utils import print_exception as pe

from linesearch import ncctools
from linesearch import build_cost_volume_structure_upsampled


def parallax2normedstructure(norm_parallax, q, B):
    """
    Convert parallax (given in pixels of motion towards epipole) to structure
    """
    h,w = norm_parallax.shape
    y,x = np.mgrid[:h,:w]
    dst = np.sqrt((x-q[0])**2 + (y-q[1])**2)
    An = (norm_parallax * B) / (norm_parallax/dst.mean() - dst / dst.mean())
    return An,dst.mean()



def build_cost_volume(images,flow,rigidity,homographies,epipoles,params):
    """
    Build cost volumes for both separate frames, where the third dimension
    of the cost volume is the structure, *not* the displacement.
    """

    BLOCKSIZE=5
    SUBSAMPLE=8
    N_RANGES = 51

    

    h,w = images[0].shape[:2]
    y,x = np.mgrid[:h,:w]

    I1 = images[1]
    I1_std,I1_mean = ncctools.std_mean(I1,5)

    if params.debug_save_frames:
        pd.plot_image(I1, '[Linesearch] Reference frame')

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
        q_new = correct_epipole(u_res,v_res,q)
        print('Q old: {}'.format(q))
        print('Q new: {}'.format(q_new))

        #
        # TODO HERE: Estimate rigidity
        #

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
            pd.plot_flow(u,v,'[Linesearch] Initial flow frame {0:01d}'.format(f))
            pd.plot_flow(u_res,v_res,'[Linesearch] Residual flow frame {0:01d}'.format(f))
            pd.plot_image(parallax, '[Linesearch] Parallax frame {0:01d}'.format(f),colorbar=True,vmin=-5,vmax=5)


    #
    # Compute backward B
    #
    # Compute valid pixels

    pv = np.logical_and(
            np.logical_and(
                np.abs(parallax_array[0])>0.1, np.abs(parallax_array[1])>0.1),
            rigidity>0)

    B_all_array = (mu_array[0] / mu_array[1]) * (parallax_array[1][pv] / parallax_array[0][pv])
    B_all_array *= (parallax_array[0][pv] / mu_array[0] - dists_normed_array[0][pv]) / (parallax_array[1][pv] / mu_array[1] - dists_normed_array[1][pv])

    print('B_all_array mean={}, median={}, std={}'.format(B_all_array.mean(), np.median(B_all_array), B_all_array.std()))

    B_b = np.median(B_all_array)
    B_array = [B_b, 1.0]

    if params.debug_save_frames:
        B_image = np.zeros_like(parallax_array[0])
        B_image[pv] = B_all_array
        pd.plot_image(B_image, '[Linesearch] Array of B_bwd', colorbar=True,vmin=-2,vmax=0)


    #
    # Compute normed structure
    #
    structure_array = []
    structure_center_removed_array = []

    for f in range(n_frames):
        q = epipoles_new_array[f]
        b = B_array[f]
        structure_normed, mu = parallax2normedstructure(parallax_array[f], q, b)
        structure_center_removed = structure_normed.copy()

        # Define an area around the epipole and set to zero.
        # Here the structure can be off by a large amount,
        # which introduces errors into the estimated range of structures.
        center_square_x_min = np.maximum(0, np.minimum(w-1, int(q[0]-10)))
        center_square_x_max = np.maximum(0, np.minimum(w-1, int(q[0]+10)))
        center_square_y_min = np.maximum(0, np.minimum(h-1, int(q[1]-10)))
        center_square_y_max = np.maximum(0, np.minimum(h-1, int(q[1]+10)))

        structure_center_removed[rigidity>0] = 0

        # If the extrema are outside the image, don't do anything.
        if center_square_x_min == w-1 or center_square_x_max == 0 or center_square_y_min == h-1 or center_square_y_max == 0:
            print('Epipole {} is too far outside of the image in frame {}'.format(q,f))
        else:
            structure_center_removed[center_square_y_min:center_square_y_max,center_square_x_min:center_square_x_max] = 0

        # Save output
        structure_array.append(structure_normed)
        structure_center_removed_array.append(structure_center_removed)


        print('Frame {}: Min structure = {}, max structure = {}'.format(f, structure_center_removed.min(), structure_center_removed.max()))

        
        if params.debug_save_frames:
            pd.plot_image(structure_normed, '[Linesearch] Normed structure in frame {}'.format(f), colorbar=True, vmin=-5,vmax=5)


    # Plot difference
    if params.debug_save_frames:
        pd.plot_image(np.abs(structure_array[0] - structure_array[1]), '[Linesearch] Structure difference', colorbar=True, vmin=0,vmax=2)


 
    #
    # Compute total structure range
    #

    # Use 1.5 times the min/max of both structure arrays with their centers removed
    structure_range_min = 1.5 * np.array(structure_center_removed_array).min()
    structure_range_max = 1.5 * np.array(structure_center_removed_array).max()
    structure_range = np.linspace(structure_range_min, structure_range_max, N_RANGES)
    print('Structure range from {} to {}: \n {}'.format(structure_range_min, structure_range_max, structure_range))


    #
    # Last step: Build actual cost volumes
    #
    costvol_array = []
    for f in range(n_frames):
        I2 = images_neighbors[f]
        I2_std,I2_mean = ncctools.std_mean(I2,BLOCKSIZE)
        q = epipoles_new_array[f]
        H = homographies[f]
        mu = mu_array[f]
        b = B_array[f]

        t0 = time.time()
        costvol = build_cost_volume_structure_upsampled.build_cost_volume(
                        I1.astype('float32'),
                        I2.astype('float32'),
                        I1_mean,I1_std, I2_mean, I2_std,
                        q.astype('float32'),
                        H.astype('float32'),
                        b,
                        structure_range.astype('float32'),
                        mu,
                        blocksize=BLOCKSIZE,
                        subsample=SUBSAMPLE)
        t1 = time.time()
        print('Frame {}: Building the cost volume took {} seconds'.format(f,t1-t0))

        costvol_array.append(costvol)


    return [costvol_array,
            structure_array,
            mu_array,
            B_array,
            epipoles_new_array,
            structure_range]



    
    raise Exception('NotImplemented')




def correct_epipole(u,v,q):
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
    #y,x = np.mgrid[:h,:w]

    xmin = int(q[0]) - 10
    xmax = int(q[0]) + 10
    ymin = int(q[1]) - 10
    ymax = int(q[1]) + 10

    y,x = np.mgrid[ymin:ymax+1,xmin:xmax+1]
    u = u[ymin:ymax+1,xmin:xmax+1]
    v = v[ymin:ymax+1,xmin:xmax+1]

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
    res = optimize.minimize(fun,x0=x0,jac=True,method='BFGS',options={'disp': True})
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



def main():
    """
    Given alignments, find best matches using a line search
    """
    print('--- Build cost volume: Starting.')

    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tempdir', type=str, default='.', help='Directory to store data files')
    args = parser.parse_args()

    # Read temp file generated by previous step in pipeline
    path_in = os.path.join(args.tempdir, '03_alignment.npz')
    data_in = np.load(path_in)

    # Define output filename and data
    data_out = dict(data_in)
    path_out = os.path.join(args.tempdir, '04_build_cost_volume.npz')

    # Check if we had an error somewhere earlier in the chain.
    if data_in['error_override']:
        print('--- overriding ---')
        # Just forward the input data
        np.savez(path_out, **data_out)
        sys.exit(0)

    #
    # Linesearch
    #
    try:
        data_return = build_cost_volume(data_in['images'],
                data_in['flow'],
                data_in['rigidity'],
                data_in['homographies'],
                data_in['epipoles'],
                data_in['params'].tolist())

        costvol = data_return[0]
        structure = data_return[1]
        mu = data_return[2]
        B = data_return[3]
        epipoles_new = data_return[4]
        structure_range = data_return[5]

    except Exception as inst:
        ## If anything bad happened, just forward initial flow.
        pe.print_exception()
        data_out['error_override'] = True
        np.savez(path_out, **data_out)
        sys.exit(0)

    # Add features to output data
    data_out['costvol'] = costvol
    data_out['structure'] = structure
    data_out['mu'] = mu
    data_out['B'] = B
    data_out['epipoles_new'] = epipoles_new
    data_out['structure_range'] = structure_range
   
    # Save output data
    print('--- Build cost volume: Saving temp file...')
    np.savez(path_out, **data_out)
    

if __name__ == '__main__':
    main()








