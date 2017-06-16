#! /usr/bin/env python2

import numpy as np

def check_homography(H, lower_right, upper_left=[0.0,0.0],output=False):
    # First check: If any entry in the homography is nan
    if np.any(np.isnan(H)):
        if output:
            print('--- [Homography check] Failed. NaN.')
        return False

    lower_right = np.array(lower_right)
    upper_left = np.array(upper_left)

    # Map the corners of the image. Check if the corners do not move too much.
    corners_in = np.array([
        [upper_left[1], lower_right[1], upper_left[1], lower_right[1]],
        [upper_left[0], upper_left[0], lower_right[0], lower_right[0]],
        [1.0, 1.0, 1.0, 1.0]])

    corners_out = H.dot(corners_in)
    corners_out = corners_out[:2,:] / corners_out[2,:]
    # Check corners differences
    shape = lower_right - upper_left
    thresh = min(shape[0],shape[1]) / 2.0
    cdiff = corners_out - corners_in[:2,:]
    corners_diff = np.linalg.norm(cdiff, axis=0)

    if any(corners_diff > thresh):
        if output:
            print('--- [Homography check] Failed. Corners are displaced too much. ---')
        return False

    #Check for images skew
    diag1 = np.linalg.norm(corners_out[:,0] - corners_out[:,3])
    diag2 = np.linalg.norm(corners_out[:,1] - corners_out[:,2])
    if max(diag1/diag2, diag2/diag1) > 1.3:
        if output:
            print('--- [Homography check] Failed. Too much skewing.')
        return False
        

    # Second check: The homography displaces the image more than half the actual image size
    if np.abs(H[0,2]) > shape[1]/2.0 or np.abs(H[1,2]) > shape[0] / 2.0:
        if output:
            print('--- [Homography check] Failed. Displacement too large.')
        return False

    # Third check: well-conditionedness.
    checkmat = np.array([
        [upper_left[1],upper_left[0],1.0],
        [upper_left[1],lower_right[0],1.0],
        [lower_right[1],upper_left[0],1.0],
        [lower_right[1],lower_right[0],1.0]]).T
    #checkmat = np.array([[shape[1],0,1],[0,shape[0],1],[shape[1],shape[0],1]]).T
    denoms = np.dot(H[2,:],checkmat)
    if np.any(np.maximum(np.abs(denoms),np.abs(1.0/denoms)) > 1.3):
        if output:
            print('--- [Homography check] Failed. Denominator too far from 1.')
        return False


    return True

