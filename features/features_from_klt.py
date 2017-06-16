#! /usr/bin/env python2

import numpy as np
import cv2

import sys,os
sys.path.append(os.environ['MRFLOW_HOME'])

# Local imports
from utils import normalized_transforms as nt

def track_klt_features(images, rigidity_thresholded, params):
    """
    Generate triple features by using KLT tracking.
    """

    # Convert images to single channel (required for tracker)
    images_bw = []
    for I in images:
        if I.ndim == 3:
            Iw = cv2.cvtColor(I.astype('float32'), cv2.COLOR_RGB2GRAY)
            Iw = (Iw*255).astype('uint8')
        else:
            Iw = (I*255).astype('uint8')
        images_bw.append(Iw)


    use_laplacian = False
    if use_laplacian:
        # Pre-filter with Laplacian
        for ni,I in enumerate(images_bw):
            If = I.astype('float32')/255.0
            L = cv2.Laplacian(If, -1)
            If = 0.5 * I + 0.5 * L
            If = (If - If.min()) / (If.max() - If.min())
            images_bw[ni] = (If*255.0).astype('uint8')


    # Extract and refine feature locations
    quality_level = 1e-6
    n_features = 10000
    min_distance = 5
    block_size = 9


    features = np.squeeze(
        cv2.goodFeaturesToTrack(images_bw[1],
                                    n_features,
                                    quality_level,
                                    min_distance,
                                    blockSize=block_size,
                                    mask=rigidity_thresholded.astype('uint8'),
                                    useHarrisDetector=False))
    features = cv2.cornerSubPix(images_bw[1],
                                    features,
                                    winSize=(5,5),
                                    zeroZone=(-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,
                                                40, 0.01))

    # Compute flow to previous and next frames
    features_prev, status_prev, error_prev = cv2.calcOpticalFlowPyrLK(
        images_bw[1], images_bw[0], features, np.zeros_like(features))
    features_next, status_next, error_next = cv2.calcOpticalFlowPyrLK(
        images_bw[1], images_bw[2], features, np.zeros_like(features))

    # There are most likely a whole lot of wrong matches.
    # Here, we apply a bunch of heuristics to filter away those wrong matches.
    #

    # First, use only features that were found in both frames
    status = status_prev[:,0] * status_next[:,0]

    # Throw out features for which the direction changes more than pi/4
    uv_prev = features - features_prev
    uv_next = features_next - features
    l_prev = np.maximum(np.linalg.norm(uv_prev, axis=1),1e-6)
    l_next = np.maximum(np.linalg.norm(uv_next, axis=1),1e-6)
    valid_direction = (uv_prev*uv_next).sum(axis=1) / (l_prev*l_next) > (1.0 / np.sqrt(2.0))

    status = status * valid_direction

    print('(MM) Features: Heuristics remove {} of {} features. {} Features remaining.'.format(
        (status==0).sum(), status.shape[0], (status==1).sum()))

    features = features[status>0,:]
    features_prev = features_prev[status>0,:]
    features_next = features_next[status>0,:]

    #
    # Second step: Remove features according to F
    #

    inliers_features_prev = nt.get_fundamental_mat_normalized(features, features_prev)[1]
    inliers_features_next = nt.get_fundamental_mat_normalized(features, features_next)[1]

    inliers_features_prev = inliers_features_prev.ravel()>0
    inliers_features_next = inliers_features_next.ravel()>0
    inliers_features_F = inliers_features_prev * inliers_features_next

    print('FundmatFilter: Removing {} of {} features.'.format(
        (inliers_features_F==0).sum(),
        inliers_features_F.size))
    rel_retain = inliers_features_F.astype('float').sum() / inliers_features_F.size

    if rel_retain < 0.5:
        print('(WW) FundmatFilter would remove more than 50%, skipping...')
        inliers_features_F[:] = True

    # We need:
    # - matches_pairwise: array of dstacked features
    # - matches_all_frames: N x 2 x F array of all corresponding features
    # - outliers_large_pairwise: outlier features (not used here)
    # - outliers_F_pairwise: outlier features according to F (pairwise)

    matches_pairwise = [
        np.dstack((features[inliers_features_F,:],features_prev[inliers_features_F,:])),
        np.dstack((features[inliers_features_F,:],features_next[inliers_features_F,:])),
        ]
    matches_all_frames = np.dstack((
        features[inliers_features_F,:],
        features_prev[inliers_features_F,:],
        features_next[inliers_features_F,:]))
    outliers_large_pairwise = [np.zeros((0,2,2)),np.zeros((0,2,2))]

    outliers_F_pairwise = [
        np.dstack((features[inliers_features_F==0,:],features_prev[inliers_features_F==0,:])),
        np.dstack((features[inliers_features_F==0,:],features_next[inliers_features_F==0,:])),
        ]

    if len(matches_all_frames) <= 4:
        print('(EE) **** Could not track enough features across frames in alignment') 
        raise Exception('NotEnoughFeaturesFound')

    print('Number of all frame matches: {}'.format(matches_all_frames.shape[0]))
    for i in range(len(matches_pairwise)):
        print('Matches in frame {}: {}'.format(i,matches_pairwise[i].shape[0]))


    return matches_pairwise, matches_all_frames, outliers_large_pairwise, outliers_F_pairwise






