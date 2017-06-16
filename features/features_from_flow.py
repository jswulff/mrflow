#! /usr/bin/env python2


import sys,os
import numpy as np
from scipy import ndimage
import cv2

sys.path.append(os.environ['MRFLOW_HOME'])
from utils import normalized_transforms as nt
from utils.imgutils import rgb2gray

def generate_features_from_flow(images, flow, occlusions, rigidity_thresholded, params):
    """
    This functions extract features from the flow.

    Additionally, it does two pre-filtering steps to remove 
    outliers and severely non-rigid points:
    1) A simple removal of the largest features is performed, and
    2) A fundamental matrix F is estimated robustly. The outliers
       in this step are taken to be outliers in the true motion as well.

    """

    # Sample features from flow maps
    featurelocations = np.zeros(flow[0][0].shape).astype('bool')
    sampling = params.feature_sampling
    featurelocations[::sampling,::sampling] = True

    # Filter out features that are in occlusions / uncertain areas
    occlusions_bwd, occlusions_fwd = occlusions
    occlusion_fwd_or_bwd = np.logical_or(occlusions_bwd>0, occlusions_fwd>0)
    featurelocations[occlusion_fwd_or_bwd] = False

    if params.featurelocations_nms > 0:
        # Refine the feature locations.
        # For this, we search for the maximum image gradient within a
        # sampling x sampling image window, and use this location.
        
        Iy,Ix = np.gradient(rgb2gray(images[1]))
        
        # This might be changed to use the actual L2 norm.
        # The minimum, however, ensures that we have a bit of gradient in
        # both directions.
        G = np.sqrt(np.minimum(Ix**2,Iy**2))
        
        # Do maximum filtering, and scale and re-scale -- this way, each
        # sampling x sampling block is set to its maximum value.
        Gm = ndimage.maximum_filter(G,size=(sampling,sampling),
                                    origin=(-sampling//2,-sampling//2))
        Gmr = cv2.resize(Gm[::sampling,::sampling],
                         dsize=(Gm.shape[1],Gm.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        
        # Now the featurelocations can be computed by comparing the gradients
        # at each pixel with the maximum blocks (each block contains the pixel
        # the maximum is coming from).
        # Also, do small filtering to ensure we have "good enough" gradients.
        
        # (the filtering G>5e-3 seems appropriate for a sampling of 10.)
        featurelocations = (G==Gmr)*(G>5e-3)
        

    # Filter out locations that overlap with rigid areas
    featurelocations = np.logical_and(featurelocations, rigidity_thresholded)

    y,x = np.mgrid[:featurelocations.shape[0],:featurelocations.shape[1]]
    x_feats = x[featurelocations]
    y_feats = y[featurelocations]

    features1 = np.c_[x_feats,y_feats]

    matches_pairwise = []
    outliers_large_pairwise = []
    outliers_F_pairwise = []
    features_all = []

    for u,v in flow:
        # Get feature motion according to flow
        uloc = u[features1[:,1],features1[:,0]]
        vloc = v[features1[:,1],features1[:,0]]

        uvloc = np.c_[uloc,vloc]
        features2 = features1 + uvloc

        #
        # Filter step 1: Remove too large features
        #
        featurelengths = np.sqrt((uvloc**2).sum(axis=1))
        data = np.c_[features1,featurelengths].astype('float64')
        data /= data.std(axis=0)

        inliers_largefilter = np.ones(features1.shape[0]) > 0

        # Save too large features (just for visualization purposes)
        features1_toolarge = features1[inliers_largefilter==0,:]
        features2_toolarge = features2[inliers_largefilter==0,:]
        # Remove too large features from set
        features1_currentframe = features1[inliers_largefilter,:]
        features2_currentframe = features2[inliers_largefilter,:]

        #
        # Filter step 2: Remove features according to homography
        #
        F,inliers_features = nt.get_fundamental_mat_normalized(features1_currentframe,features2_currentframe)
        inliers_features = inliers_features.ravel()>0
        print('FundmatFilter: Removing {} of {} features.'.format((inliers_features==0).sum(),inliers_features.size))
        rel_retain = inliers_features.astype('float').sum() / inliers_features.size

        if rel_retain < 0.5:
            print('(WW) FundmatFilter would remove more than 50%, skipping...')
            inliers_features[:] = True

        # And again, save outliers
        features1_fundmat_outlier = features1_currentframe[inliers_features==False,:]
        features2_fundmat_outlier = features2_currentframe[inliers_features==False,:]

        features1_currentframe = features1_currentframe[inliers_features==True, :]
        features2_currentframe = features2_currentframe[inliers_features==True, :]


        matches_pairwise.append(np.dstack((features1_currentframe,features2_currentframe)))

        outliers_large_pairwise.append(
                np.dstack((
                    features1_toolarge,
                    features2_toolarge)))

        outliers_F_pairwise.append(
                np.dstack((
                    features1_fundmat_outlier,
                    features2_fundmat_outlier)))

    # After loop, compute matches that are valid across all frames
    matches_all_frames = pairwise_to_global(matches_pairwise,len(images))

    if len(matches_all_frames) <= 4:
        print('(EE) **** Could not track enough features across frames in alignment') 
        raise Exception('NotEnoughFeaturesFound')

    print('Number of all frame matches: {}'.format(matches_all_frames.shape[0]))
    for i in range(len(matches_pairwise)):
        print('Matches in frame {}: {}'.format(i,matches_pairwise[i].shape[0]))



    return matches_pairwise, matches_all_frames, outliers_large_pairwise, outliers_F_pairwise


def pairwise_to_global(matches,n_images):
    """
    Utility function to convert pairwise matches
    to global matches.
    """
    # matches[:][:,:,0] is always the reference frame.
    candidates_ar = matches[0][:,:,0]
    candidates_list = [tuple(c) for c in candidates_ar]
    candidates_set = set(candidates_list)
    for m in matches[1:]:
        candidate_query = set([tuple(c) for c in m[:,:,0]])
        candidates_set = candidates_set & candidate_query
    
    # Now candidates_set includes the coordinates of features in the reference frame
    # that exist in every pair. Next, fill with actual matches.
    matches_all_frames = np.zeros((len(candidates_set),2,n_images))
    
    # Fill reference coordinates
    inds = [candidates_list.index(c) for c in candidates_set]
    if len(inds) < 4:
        return np.array([])

    matches_all_frames[:,:,0] = matches[0][np.array(inds),:,0]
    
    for i, m in enumerate(matches):
        list_reference = [tuple(c) for c in m[:,:,0]]
        inds = [list_reference.index(c) for c in candidates_set]
        matches_all_frames[:,:,i+1] = m[np.array(inds),:,1]
    return matches_all_frames

