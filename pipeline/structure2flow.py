#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

sys.path.append(os.environ['MRFLOW_HOME'])

# Local imports
from utils import flow_homography as fh
from utils import compute_figure


def structure_to_flow(structure, q, mu, H, b, params):
    """Convert structure back to full flow

    """

    #
    # Step 1: Convert structure to parallax
    #
    y,x = np.mgrid[:structure.shape[0],:structure.shape[1]]
    dists = np.sqrt((x-q[0])**2 + (y-q[1])**2)
    # Normalized distances
    n = dists / mu;
    # Convert to parallax
    parallax = (b * structure * n) / (b * structure/mu - 1)

    # Step 2: Convert parallax to residual flow
    xv = (q[0] - x) / np.maximum(1e-3,dists)
    yv = (q[1] - y) / np.maximum(1e-3,dists)
    u_res = xv * parallax
    v_res = yv * parallax

    # step 3: Convert residual flow to full flow
    u, v = fh.get_full_flow(u_res,v_res,H,shape=u_res.shape)

    return u, v


def combine_flow(structure, flow_init, rigidity,
        q_ar, mu_ar, H_ar, b_ar, images, params, method='simple'):
    """ Merge rigid and non-rigid flows.
    """

    rigidity, u,v = combine_flow_simple(
                structure,
                flow_init,
                rigidity,
                q_ar, mu_ar, H_ar, b_ar,
                images,
                params)

    return rigidity, u,v


def combine_flow_simple(structure, flow_init, rigidity,
        q_ar, mu_ar, H_ar, b_ar, images, params):
    """ Simple flow computation purely based on rigidity map.
    """

    u_forward, v_forward = structure_to_flow(
            structure, q_ar[1], mu_ar[1], H_ar[1], b_ar[1], params)


    if params.debug_compute_figure == 94:
        compute_figure.plot_figure_video_pasted_example(
            rigidity, flow_init, [u_forward,v_forward])
        sys.exit(1)


    u_forward[rigidity==0] = flow_init[0][rigidity==0]
    v_forward[rigidity==0] = flow_init[1][rigidity==0]

    if params.debug_compute_figure == 95:
        compute_figure.plot_figure_95(
            images, rigidity, structure, flow_init,
            [u_forward, v_forward])
        sys.exit(1)


    return rigidity, u_forward, v_forward

