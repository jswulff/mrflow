#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:08:24 2016

@author: jonas
"""


"""
A bunch of robust functions. Each generator function returns the function itself
and its derivative.

As per standard convention, the parameter for all the functions is x^2, meaning
that the derivative is dF/(dx^2).

For all functions requiring a sigma, it can be set to 0. In this case, the sigma
is automatically computed using the MAD.
"""

import numpy as np

def _mad(x):
    return np.median(np.abs(x-np.median(x)))

def generate_charbonnier(eps=0, a=0.5):
    if a == 0.5:
        if eps == 0:
            sigma_l = lambda x : np.maximum(1e-6, 1.4826 * _mad(np.c_[np.sqrt(x),-np.sqrt(x)]))
            f = lambda x : np.sqrt(x+sigma_l(x)**2)
            df = lambda x : 1.0 / (2*np.sqrt(x + sigma_l(x)**2))
        else:
            f = lambda x : np.sqrt(x+eps**2)
            df = lambda x : 1.0 / (2*np.sqrt(x + eps**2))
    else:
        if eps == 0:
            sigma_l = lambda x : np.maximum(1e-6, 1.4826 * _mad(np.c_[np.sqrt(x),-np.sqrt(x)]))
            f = lambda x : (x+sigma_l(x)**2)**a
            df = lambda x : a * (x+sigma_l(x)**2)**(a-1)
        else:
            f = lambda x : (x+eps**2)**a
            df = lambda x : a * (x+eps**2)**(a-1)
    return f,df
    
def generate_quadratic():
    f = lambda x : x
    df = lambda x : np.ones_like(x)
    return f,df
    
def generate_lorentzian(sigma=0):
    if sigma > 0:
        f = lambda x : np.log(1.0+ x / (sigma**2))
        df = lambda x : 1.0/ (sigma**2 + x)
    else:
        sigma_l = lambda x : np.maximum(1e-6, 1.4826 * _mad(np.c_[np.sqrt(x),-np.sqrt(x)]))
        # f = lambda x : sigma_l(x)**2 * np.log(1.0 + x / (sigma_l(x) ** 2))
        # df = lambda x : sigma_l(x)**2 / ( x + sigma_l(x) ** 2)
        f = lambda x : sigma_l(x) * np.log(1.0 + 0.5 * x / (sigma_l(x) ** 2))
        df = lambda x : sigma_l(x) / ( x + 2*sigma_l(x) ** 2)
    return f,df
    
def generate_geman_mcclure(sigma=0):
    if sigma > 0:
        f = lambda x : sigma * x / (x + sigma**2)
        df = lambda x : sigma**3 /  ((x +  sigma**2)**2)
    else:
        sigma_l = lambda x : np.maximum(1e-6, 1.4826 * _mad(np.c_[np.sqrt(x),-np.sqrt(x)]))
        f = lambda x : sigma_l(x) * x / (x + sigma_l(x)**2)
        df = lambda x : sigma_l(x)**3 /  ((x +  sigma_l(x)**2)**2)
    return f,df




