import numpy as np
import cv2

def get_fundamental_mat_normalized(p1,p2):
    F_, inliers = cv2.findFundamentalMat(p1.astype('float32'), p2.astype('float32'), method=cv2.LMEDS)
    return F_,inliers

def get_perspective_transform_normalized(p1,p2):
    """Get least squares estimate of homography.
    """
    H = find_homography_normalized(p1,p2,robust=False)
    return H

def find_homography_normalized(p1,p2,robust=True):
    """Get best estimate of homography.

    Parameters:
    robust : bool, optional
        If set to True (default), use LMedS estimation. If set to False,
        use least squares.

    """
    p1,p2,N1,N2 = normalize_points(p1,p2)
    if robust:
        method=cv2.LMEDS
    else:
        method=0
    H_, inliers = cv2.findHomography(p1.astype('float32'),
            p2.astype('float32'),
            method=method)
    H = np.linalg.inv(N2).dot(H_.astype('float')).dot(N1)
    H /= H[2,2]
    return H

#
# Two simple wrappers to use the same interface
#
def get_perspective_transform_unnormalized(p1,p2):
    """Get least squares estimate of homography without normalization.
    """
    H = find_homography_unnormalized(p1,p2,robust=False)
    return H

def find_homography_unnormalized(p1,p2,robust=True):
    """Get best estimate of homography without normalization.

    Parameters:
    robust : bool, optional
        If set to True (default), use LMedS estimation. If set to False,
        use least squares.

    """
    if robust:
        method=cv2.LMEDS
    else:
        method=0
    H, inliers = cv2.findHomography(p1.astype('float32'),
            p2.astype('float32'),
            method=method)
    return H




def old_get_perspective_transform_normalized(p1,p2):
    """
    A small wrapper around cv2.getPerspectiveTransform, with normalization of
    point locations.

    """

    return cv2.getPerspectiveTransform(p1,p2)

    mu1 = p1.mean(axis=0)
    std1 = p1.std(axis=0)
    mu2 = p2.mean(axis=0)
    std2 = p2.std(axis=0)

    p1_ = (p1 - mu1) / std1
    p2_ = (p2 - mu2) / std2

    H_ = cv2.getPerspectiveTransform(p1_,p2_)
    A1 = np.array([[1.0/std1[0], 0.0, -mu1[0]/std1[0]],
                   [0, 1.0/std1[1], -mu1[1]/std1[1]],
                   [0,0,1.0]])
    A2inv = np.array([[std2[0], 0.0, mu2[0]],
                   [0, std2[1], mu2[1]],
                   [0,0,1.0]])
    H = A2inv.dot(H_).dot(A1)
    return H

def old_find_homography_normalized(p1,p2):
    """
    A small wrapper around cv2.getPerspectiveTransform, with normalization of
    point locations.

    """

    return cv2.findHomography(p1,p2,method=cv2.LMEDS)[0]

    mu1 = p1.mean(axis=0)
    std1 = p1.std(axis=0)
    mu2 = p2.mean(axis=0)
    std2 = p2.std(axis=0)

    p1_ = (p1 - mu1) / std1
    p2_ = (p2 - mu2) / std2

    H_ = cv2.findHomography(p1_,p2_,method=cv2.LMEDS)[0]
    A1 = np.array([[1.0/std1[0], 0.0, -mu1[0]/std1[0]],
                   [0, 1.0/std1[1], -mu1[1]/std1[1]],
                   [0,0,1.0]])
    A2inv = np.array([[std2[0], 0.0, mu2[0]],
                   [0, std2[1], mu2[1]],
                   [0,0,1.0]])
    H = A2inv.dot(H_).dot(A1)
    return H


def old_get_fundamental_mat_normalized(p1,p2,use_ransac=False):
    """
    A small wrapper around cv2.getFundamentalMat, with normalization of
    point locations.

    """
    mu1 = p1.mean(axis=0)
    std1 = p1.std(axis=0)
    mu2 = p2.mean(axis=0)
    std2 = p2.std(axis=0)

    p1_ = (p1 - mu1) / std1
    p2_ = (p2 - mu2) / std2

    if use_ransac:
        F_, inliers = cv2.findFundamentalMat(p1_,p2_,method=cv2.FM_RANSAC,param1=4 * 2.0/(std1+std2).mean())
    else:
        F_,inliers = cv2.findFundamentalMat(p1_,p2_,method=cv2.FM_LMEDS)
    A1 = np.array([[1.0/std1[0], 0.0, -mu1[0]/std1[0]],
                   [0, 1.0/std1[1], -mu1[1]/std1[1]],
                   [0,0,1.0]])
    A2 = np.array([[1.0/std2[0], 0.0, -mu2[0]/std2[0]],
                   [0, 1.0/std2[1], -mu2[1]/std2[1]],
                   [0,0,1.0]])
    F = A2.T.dot(F_).dot(A1)
    #F = A2inv.dot(H_).dot(A1)
    return F,inliers


#
# Normalization functions
#
def normalize_points(p1,p2,uniform=False):
    """
    Normalize points, either per-frame or uniformly.
    """
    p1 = p1.astype('float')
    p2 = p2.astype('float')
    
    if uniform:
        p = np.r_[p1,p2]
        mean = p.mean(axis=0)
        p -= mean
        p1 -= mean
        p2 -= mean
        s = np.linalg.norm(p,axis=1).mean() / np.sqrt(2.0)
        p1 /= s
        p2 /= s
        
        s1 = s
        s2 = s
        p1mean = mean
        p2mean = mean
    else:
        p1mean = p1.mean(axis=0)
        p2mean = p2.mean(axis=0)
        p1 -= p1mean
        p2 -= p2mean
        s1 = np.linalg.norm(p1,axis=1).mean() / np.sqrt(2.0)
        s2 = np.linalg.norm(p2,axis=1).mean() / np.sqrt(2.0)
        p1 /= s1
        p2 /= s2

    N1 = np.array([[1.0 / s1, 0.0, -p1mean[0] / s1],
                  [0.0, 1.0 / s1, -p1mean[1] / s1],
                  [0.0, 0.0, 1.0]])
    N2 = np.array([[1.0 / s2, 0.0, -p2mean[0] / s2],
                  [0.0, 1.0 / s2, - p2mean[1] / s2],
                  [0.0, 0.0, 1.0]])
    
    return p1,p2,N1,N2

        


