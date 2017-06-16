#! /usr/bin/env python2

import sys,os

import numpy as np
from scipy import special # for ive

from utils.spyder_debug import *
from utils.imgutils import img_as_ubyte

# Try to import external packages
HAVE_DENSECRF = False
try:
    from densecrf import pydensecrf
    HAVE_DENSECRF = True
except:
    pydensecrf = None

HAVE_TRWS = False
try:
    from extern.eff_trws import eff_trws
    HAVE_TRWS = True
except:
    Eff_TRWS = False



def get_unaries_rigid(x, y, u_res, v_res, qx, qy ,sigma=1.0,prior_rigid=0.5, prior_nonrigid=0.5):
    """
    Compute the probability that point (x,y) with flow (u,v) is rigid (ie, points to q).
    
    p = (x,y), p' = (x+u,y+v), q = FoE.
    
    Assumptions:
    - A noise around the feature match with Gaussian normal distribution
    - A uniform distribution over the space for all rigid points
    - Uninformative prior on rigid/non-rigid.
    """
    
    # Compute the distance of match and the angle between match and FoE.
    dist = np.sqrt(u_res**2 + v_res**2)
    
    ang_foe = np.arctan2(qy-y,qx-x)
    ang_uv = np.arctan2(v_res,u_res)
    ang = ang_uv - ang_foe
   
    # Compute probability that p' points towards epipole
    dist_from_line = dist * np.sin(ang)
    
    p = 1.0 / np.sqrt(2*np.pi*sigma**2) * np.exp( - dist_from_line**2 / (2*sigma**2))
    
    # Normalization constant
    # Note that we use special.ive = exp(-x) * special.iv(x) for numerical stability.
    c = np.sqrt(2 * np.pi) / sigma * special.ive(0,dist**2 / (sigma**2 * 4))
            
    # Probability that point is rigid is given as
    # p(rigid) = p(point|rigid) / ( p(point|rigid) + p(point|nonrigid)).
    prob = prior_rigid * p / (prior_rigid * p + prior_nonrigid * c / (2*np.pi))
    return prob



def infer_densecrf(I, unaries, gaussian_sigma=5, gaussian_weight=1, bilateral_sigma_spatial=11, bilateral_sigma_col=1, bilateral_weight=10):
    """ DenseCRF inference
    """

    if not HAVE_DENSECRF:
        print('********************************************************************************')
        print('(EE) DenseCRF code not found.')
        print('********************************************************************************')
        sys.exit(0)

    I_ = img_as_ubyte(I)
    n_labels = 2

    unaries_ = -np.log(unaries).transpose(2,0,1).reshape((2,-1)).astype('float32').copy('c')

    densecrf = pydensecrf.pyDenseCRF2D(I.shape[1],I.shape[0],n_labels)
    densecrf.setUnaryEnergy(unaries_)

    # Parameters: sx, sy, weight_gaussian. This is independent of the color.
    densecrf.addPairwiseGaussian_Potts(gaussian_sigma,gaussian_sigma,gaussian_weight)

    # Parameters: x,y,r,g,b
    densecrf.addPairwiseBilateral_Potts(bilateral_sigma_spatial,
            bilateral_sigma_spatial,
            bilateral_sigma_col,
            bilateral_sigma_col,
            bilateral_sigma_col,
            I_,
            bilateral_weight)

    result = np.zeros((I.shape[0],I.shape[1]),dtype='int32')
    densecrf.compute_map(result)

    return result > 0


def infer_mrf(I, unaries, lambd=1.1):
    """ MRF inference using weighted neighbor potentials.
    """
    if not HAVE_TRWS:
        print('********************************************************************************')
        print('(EE) TRWS code not found.')
        print('********************************************************************************')
        sys.exit(0)

    h,w,nlabels = unaries.shape

    # Compute edge weights from image
    weights_e, weights_n, weights_ne, weights_se = get_image_weights(I)

    unaries_ = ((-unaries) * 1000).astype('int32').copy('C')
    #unaries_ = -np.log(unaries).astype('int32')

    TRUNCATION=1
    NEIGHBORHOOD=8

    TRWS = eff_trws.Eff_TRWS(w,h,nlabels, truncation=TRUNCATION, neighborhood=NEIGHBORHOOD)
    labels_out = np.zeros((h,w),dtype=np.int32)

    # Do the optimization.
    # Note that the use_trws flag is set to false, so we use a standard alpha/beta swap.
    # For some reason, TRWS does not always produce a good labelling.
    TRWS.solve(unaries_,
            int(lambd*1000),
            labels_out, 
            weights_horizontal=weights_e,
            weights_vertical=weights_n,
            weights_ne=weights_ne,
            weights_se=weights_se,
            use_trws=False,
            effective_part_opt=True)

    return labels_out


    

def get_image_weights(I):
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


