# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:12:14 2016

@author: jonas
"""

#! /usr/bin/env python2

import numpy as np

plt=None
try:
    from matplotlib import pyplot as plt
except:
    pass

def imshow(x, **kwargs):
    if plt is None:
        return
    plt.figure()
    plt.imshow(x, **kwargs)
    plt.pause(1000)
    
def close():
    if plt is None:
        return    
    plt.close()
    
def plot(x, **kwargs):
    if plt is None:
        return    
    plt.figure()
    plt.plot(x, **kwargs)
    plt.pause(1000)
    