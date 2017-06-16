#! /usr/bin/env python2

import numpy as np
import cv2





def computeOcclusionsFromConsistency(flow, backflow, threshold=7.0):
    """ Compute occlusions from backward-forward consistency.

    Parameters
    ----------
    flow : array of array of ndimages
      Array of flows, so that flow[0][0] is the horizontal component of the
      backward flow, flow[1][1] the vertical component of the forward flow and
      so on.
    backflow : array of array of ndimages
      Similar to flow, but here each ndimage contains the motion towards the
      reference frame.
      Example: flow[0] is the flow from t-1 to t.
    threshold : float
      The relative threshold above which the flow is considered to be invalid
      or an occlusion.
    """

    h,w = flow[0][0].shape

    def getWarpedError(uf,vf,ub,vb):
        y,x = np.mgrid[:uf.shape[0],:uf.shape[1]]
        u_warped = cv2.remap(ub, 
                            (x+uf).astype('float32'),
                            (y+vf).astype('float32'),
                            interpolation=cv2.INTER_LINEAR)
        v_warped = cv2.remap(vb,
                            (x+uf).astype('float32'),
                            (y+vf).astype('float32'),
                            interpolation=cv2.INTER_LINEAR)

        valid = (y+vf >= 0) * (y+vf < y.shape[0]) * (x+uf >= 0) * (x+uf < x.shape[1])
        err = np.sqrt((u_warped+uf)**2 + (v_warped+vf)**2)

        return err, valid==0

    error_backward, invalid_backward = getWarpedError(
        flow[0][0],
        flow[0][1],
        backflow[0][0],
        backflow[0][1])
    error_forward, invalid_forward = getWarpedError(
        flow[1][0],
        flow[1][1],
        backflow[1][0],
        backflow[1][1])

    thresh_bwd = threshold
    thresh_fwd = threshold

    # old way to compute occlusions
    occ_backward = np.logical_or(error_backward > thresh_bwd, invalid_backward)
    occ_forward = np.logical_or(error_forward > thresh_fwd, invalid_forward)

    # Properly add invalid regions
    invalid_both = invalid_backward * invalid_forward
    invalid_only_forward = np.logical_and(invalid_forward, invalid_backward==0)
    invalid_only_backward = np.logical_and(invalid_backward, invalid_forward==0)
    occ_backward[invalid_only_forward] = 0
    occ_forward[invalid_only_backward] = 0


    occ_both = occ_backward * occ_forward

    return occ_backward, occ_forward, occ_both






