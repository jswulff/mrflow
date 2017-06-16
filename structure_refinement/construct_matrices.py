#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:59:32 2016

@author: jonas
"""

import numpy as np
from scipy import sparse

from utils import matrix_from_filter

def construct_gradient_matrix(shape_image, return_separate=False):
    h,w = shape_image
    Dx = matrix_from_filter.matrix_from_filter(np.array([[0,-1.0,1.0]]),shape=shape_image)
    Dy = matrix_from_filter.matrix_from_filter(np.array([[0,-1.0,1.0]]).T,shape=shape_image)

    if return_separate:
        return Dx.tocsr(), Dy.tocsr()
    else:
        G = sparse.vstack((Dx,Dy)).tocsr()
        return G
    
def construct_2ndorder_matrix(shape_image, return_separate=False):
    h,w = shape_image
    Dxx = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]),shape=shape_image)
    Dyy = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]).T,shape=shape_image)

    #filt_xy = np.array([[0.0,1.0,-1.0],[0.0,-1.0,1.0],[0.0,0.0,0.0]])
    #Dxy = matrix_from_filter.matrix_from_filter(filt_xy, shape=shape_image)
    #G = sparse.vstack((Dxx,Dxy,Dyy)).tocsr()

    if return_separate:
        return Dxx.tocsr(), Dyy.tocsr()
    else:
        G = sparse.vstack((Dxx,Dyy)).tocsr()
        return G


def construct_2ndorder_matrix_full(shape_image):
    h,w = shape_image
    Dxx = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]),shape=shape_image)
    Dyy = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]).T,shape=shape_image)

    #filt_xy = np.array([[0.0,1.0,-1.0],[0.0,-1.0,1.0],[0.0,0.0,0.0]])
    filt_xy = np.array([[0.0,0.0,0.0],[0.0,1.0,-1.0],[0.0,-1.0,1.0]])
    Dxy = matrix_from_filter.matrix_from_filter(filt_xy, shape=shape_image)
    G = sparse.vstack((Dxx,Dxy,Dyy)).tocsr()
    return G

def construct_2ndorder_matrix_div(shape_image):
    h,w = shape_image
    Dxx = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]),shape=shape_image)
    Dyy = matrix_from_filter.matrix_from_filter(np.array([[1.0,-2.0,1.0]]).T,shape=shape_image)

    #filt_xy = np.array([[0.0,1.0,-1.0],[0.0,-1.0,1.0],[0.0,0.0,0.0]])
    filt_xy = np.array([[0.0,0.0,0.0],[0.0,1.0,-1.0],[0.0,-1.0,1.0]])
    Dxy = matrix_from_filter.matrix_from_filter(filt_xy, shape=shape_image)
    G = sparse.vstack((Dxx,Dxy,Dyy)).T.tocsr()
    return G



