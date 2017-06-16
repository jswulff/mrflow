#! /usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt

def structure2image(structure, rigidity_refined,cmap='hot', structure_min=None, structure_max=None):
    if structure_min is None:
        structure_min = np.percentile(structure[rigidity_refined==1].ravel(), 2)
    if structure_max is None:
        structure_max = np.percentile(structure[rigidity_refined==1].ravel(), 98)
    Istructure = (structure - structure_min) / max(1e-6,structure_max-structure_min)
    Istructure = np.clip(Istructure,0,1)

    cm = plt.get_cmap(cmap)
    Istructure = cm(Istructure)[:,:,:3]*255.0
    Istructure[:,:,0][rigidity_refined==0] = 128
    Istructure[:,:,1][rigidity_refined==0] = 0
    Istructure[:,:,2][rigidity_refined==0] = 128
    return Istructure.astype('uint8')
   

