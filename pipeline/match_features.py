#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

sys.path.append(os.environ['MRFLOW_HOME'])


from features import features_from_flow, features_from_klt


def generate_features(images, flow, occlusions, rigidity_thresholded, params):
    """ Extract features.
    This is a simple wrapper that decides whether the features should be
    extracted from the flow, or computed as KLT features.
    """
    if params.use_klt:
        retvals = features_from_klt.track_klt_features(images, rigidity_thresholded,params)

        # Sanity check - if the tracked features are too imbalanced, ie
        # too concentrated on one or another side, use features from computed
        # flow instead.
        matches_all_frames = retvals[1]
        feature_upper = np.percentile(matches_all_frames[:,:,0],95,axis=0)
        feature_lower = np.percentile(matches_all_frames[:,:,0],5,axis=0)
        feature_stds = feature_upper - feature_lower
        h,w = images[0].shape[:2]
        if (feature_stds[0] < 0.4 * w) or (feature_stds[1] < 0.4 * h):
            print('(EE) ********************')
            print('(EE) Features are too imbalanced. Using initial flow instead.')
            print('(EE) ********************')
            retvals = features_from_flow.generate_features_from_flow(
                images, flow, occlusions, rigidity_thresholded, params)

    else:
        retvals = features_from_flow.generate_features_from_flow(
            images, flow, occlusions, rigidity_thresholded, params)

    return retvals




