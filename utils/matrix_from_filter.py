#! /usr/bin/env python2

import numpy as np
from scipy import sparse

"""
Generate sparse matrix from given filter, so that (for an image)
    M.dot(I.ravel()) = correlate(I, filter).

It is similar to make_imfilter_mat from Stefan Roth's MATLAB code.
"""


def matrix_from_filter(filt, shape, spformat='csr'):
    """ Generate sparse matrix from given filter.

    Matrix is generated so that for an image I
        M.dot(I.ravel()) = correlate(I, filter).

    WARNING: Currently, it only supports 3x3 filters!
    
    It is similar to make_imfilter_mat from Stefan Roth's MATLAB code.
    At the boundaries, the last valid value is replicated.

    Parameters
    ----------
    filt : array_like, float
        Correlation kernel. For now it is assumed that the side has odd
        length, and is centered at the center pixel.
    shape : (h,w) tuple
        Shape of original image. The resulting sparse matrix will have
        shape (h*w, h*w).
    format : string, optional.
        Denotes the format of the returned sparse matrix. Can be
        One of ('lil', 'csr', 'csc'). Default: 'csr'.


    Returns
    -------
    mat : sparse array_like.
        Return matrix.

    """
    h,w = shape
    y,x = np.mgrid[:h,:w]
    y = y.flatten()
    x = x.flatten()
    
    sl = h*w

    # Construct basic inner matrix
    filtshape = filt.shape
    offset_x = filtshape[1]//2
    offset_y = filtshape[0]//2
    
    # Compute "side-diagonal mask". For each entry in the filter,
    # this mask contains the side diagonals this entry corresponds
    # to.
    fy,fx = np.mgrid[:filtshape[0],:filtshape[1]]
    
    
    fx_filtered = fx[filt!=0]-offset_x
    fy_filtered = fy[filt!=0]-offset_y
    
    sdmask = (fy-offset_y)*w + fx-offset_x
    
    diags = sdmask[filt!=0].tolist()
    data = np.tile(filt[filt!=0],[w*h,1]).T
    
    filtvals = filt[filt!=0]
    
    # For each entry in the filter, compute when it leaves the frame.
    diags_prev = [d for d in diags]
    for ind, diag in enumerate(diags_prev):
        relco_x = fx_filtered[ind]
        relco_y = fy_filtered[ind]
        #print('Index = {}, Diagonal = {}, rx = {}, ry = {}'.format(ind, diag, relco_x, relco_y))
        
        inv_x_smaller = x + relco_x < 0
        inv_x_larger = x + relco_x >= w
        inv_y_smaller = y + relco_y < 0
        inv_y_larger = y + relco_y >= h
        
        
        
        #
        # The issue here is that these "invalid" indices are always the indices of the *center* pixel
        # (i.e. if the sidelength is 5, then at center pixel index = 5 the right pixel should be disabled,
        # and at center pixel index = 6 the left pixel should be disabled.)
        # This is NOT equivalent to the indices on the diagonals.


        #
        # Process out-of-frame, left
        #
        if np.any(inv_x_smaller):
            diag_update = diag - relco_x
            if diag_update in diags:
                ind_update = diags.index(diag_update)
            else:
                diags.append(diag_update)
                ind_update = len(diags)-1
                data = np.vstack((data, np.zeros((h*w))))
            inv_nz = np.nonzero(inv_x_smaller)[0]
            data[ind_update,inv_nz] += filtvals[ind]
            if diag < 0:
                data[ind,inv_nz+diag] = 0
            else:
                data[ind,inv_nz] = 0
            
            
        # out-of-frame, right
        if np.any(inv_x_larger):
            diag_update = diag - relco_x # 1 #relco_x
            if diag_update in diags:
                ind_update = diags.index(diag_update)
            else:
                diags.append(diag_update)
                ind_update = len(diags)-1
                data = np.vstack((data, np.zeros((h*w))))
            inv_nz = np.nonzero(inv_x_larger)[0]
            data[ind_update,inv_nz] += filtvals[ind]
            if diag < 0:
                data[ind,inv_nz+diag] = 0
            else:
                data[ind,inv_nz] = 0
            
        if np.any(inv_y_larger):
            diag_update = diag - w *relco_y
            if diag_update in diags:
                ind_update = diags.index(diag_update)
            else:
                diags.append(diag_update)
                ind_update = len(diags)-1
                data = np.vstack((data, np.zeros((h*w))))
            inv_nz = np.nonzero(inv_y_larger)[0]
            data[ind_update,inv_nz] += filtvals[ind]
                               
        if np.any(inv_y_smaller):
            diag_update = diag - w *relco_y
            if diag_update in diags:
                ind_update = diags.index(diag_update)
            else:
                diags.append(diag_update)
                ind_update = len(diags)-1
                data = np.vstack((data, np.zeros((h*w))))
            inv_nz = np.nonzero(inv_y_smaller)[0]
            data[ind_update,inv_nz] += filtvals[ind]
            


    # Update corners
    if 0 in diags:
        maindiag = diags.index(0)
    else:
        diags.append(0)
        maindiag = len(diags)-1
        data = np.vstack((data, np.zeros((h*w))))
                
    if filt.shape[0]>1 and filt.shape[1]>1:
        data[maindiag,0] += filt[0,0]
        data[maindiag,w-1] += filt[0,-1]
        data[maindiag,-w] += filt[-1,0]
        data[maindiag,-1] += filt[-1,-1]
        
       
    mat = sparse.diags(data,diags,shape=(sl,sl),format=spformat)
    return mat
