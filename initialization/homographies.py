#! /usr/bin/env python2
import numpy as np
import chumpy as ch
import time

from scipy.linalg import sqrtm
from scipy import optimize

import cv2

from robust_fitting import ransac, lmeds

from utils import normalized_transforms as nt
from utils.check_homography import check_homography


def compute_initial_homographies(matches_all_frames, ransacReprojThreshold=1.0):
    """ Compute a set of initial homographies from the matches given in matches_all_frames
    """

    def estimate_model(X):
        homographies = []
        for frame in range(1,X.shape[2]):
            pt1 = X[:,:,0]
            pt2 = X[:,:,frame]
            if pt1.shape[0] < 4:
                # print('Too few points')
                return None
            elif pt1.shape[0] == 4:
                H = nt.get_perspective_transform_normalized(pt1.astype('float32'),pt2.astype('float32'))
            else:
                H = nt.find_homography_normalized(pt1.astype('float32'),
                        pt2.astype('float32'))
                if H is None:
                    # print('Could not find homography in the end')
                    return None
            if not check_homography(H, [436,1024]):
                # print('Homography check failed.')
                return None

            homographies.append(H)

        model = np.array([H.ravel() for H in homographies])
        return model

    def estimate_inliers(X, model):
        dists = np.zeros((X.shape[0],X.shape[2]-1))
        for frame in range(X.shape[2]-1):
            H = model[frame,:].reshape((3,3))
            Hinv = np.linalg.inv(H).astype('float32')
            H = H.astype('float32')
            pt1 = X[:,:,0].astype('float32')
            pt2 = X[:,:,frame+1].astype('float32')
            pt1_projected = cv2.perspectiveTransform(pt1.reshape((-1,1,2)),H).squeeze()
            pt2_projected = cv2.perspectiveTransform(pt2.reshape((-1,1,2)),Hinv).squeeze()
            errors1 = np.linalg.norm(pt2 - pt1_projected,axis=1)
            errors2 = np.linalg.norm(pt1 - pt2_projected,axis=1)
            dists[:,frame] = (errors1+errors2)/2.0
        inliers = np.all(dists<=ransacReprojThreshold,axis=1)
        return inliers

    homographies_model, inliers_ = ransac.estimate(matches_all_frames,
                                                   4,
                                                   estimate_model,
                                                   estimate_inliers,
                                                   p_outlier=0.8,
                                                   do_print=True,
                                                   min_iters=10000,
                                                   seed=12345)

    if homographies_model is None:
        print('(EE) **** Could not compute a homography')
        raise Exception('NoHomographyFound')
    else:
        print('(MM) Number of inliers: {}'.format(inliers_.sum()))
        
    homographies_init = [h.reshape((3,3)) for h in homographies_model]

    return homographies_init





def refine_homographies(matches_all_frames_, homographies_init,shape,error_type=0):
    """ Refine initial homographies to make sure they correspond as much to a true plane as possible.
    """
    def chumpy_get_H(p1,p2):
        """ Compute differentiable homography from p1 to p2.
        """
        N = p1.shape[0]
        A1 = ch.vstack(( ch.zeros((3, N)), -p1.T, -ch.ones((1, N)), p2[:,1] * p1[:,0], p2[:,1] * p1[:,1], p2[:,1] )).T
        A2 = ch.vstack(( p1.T, ch.ones((1,N)), ch.zeros((3, N)), -p2[:,0] * p1[:,0], -p2[:,0] * p1[:,1], -p2[:,0] )).T
        A = ch.vstack((A1,A2))

        U,S,V = ch.linalg.svd(A.T.dot(A))
        H_new = V[-1,:].reshape((3,3))
        return H_new


    def chumpy_compute_error(p1,p2,H,Mcor,error=0,return_sigma=False,sigma_precomputed=None):
        """Compute deviation of p1 and p2 from common epipole, given H.
        """
        # Warp p2 estimates with new homography
        p2_new_denom = H[2,0] * p2[:,0] + H[2,1] * p2[:,1] + H[2,2]
        p2_new_x = (H[0,0] * p2[:,0] + H[0,1] * p2[:,1] + H[0,2]) / p2_new_denom
        p2_new_y = (H[1,0] * p2[:,0] + H[1,1] * p2[:,1] + H[1,2]) / p2_new_denom
        p2_new = ch.vstack((p2_new_x,p2_new_y)).T

        # Compute current best FoE
        u = p2_new[:,0] - p1[:,0]
        v = p2_new[:,1] - p1[:,1]
        T = ch.vstack((-v,u,v*p1[:,0] - u * p1[:,1])).T
        U,S,V = ch.linalg.svd(Mcor.dot(T.T.dot(T)).dot(Mcor))
        qv = Mcor.dot(V[-1,:])
        q = qv[:2]/qv[2]

        d = T.dot(qv)

        # Robust error norm
        if error == 0:
            d = d**2
        elif error == 1:
            sigma = np.median(np.abs(d() - np.median(d())))
            d = ch.sqrt(d**2 + sigma**2)
        elif error == 2:
            # Geman-McClure
            sigma = np.median(np.abs(d() - np.median(d()))) *1.5
            d = d**2 / (d**2 + sigma)
        elif error == 3:
            # Geman-McClure, corrected
            if sigma_precomputed is not None:
                sigma = sigma_precomputed
            else:
                sigma = 1.426 * np.median(np.abs(d() - np.median(d()))) * np.sqrt(3.0)

            d = d**2 / (d**2 + sigma**2)
            # Correction
            d = d * sigma

        elif error == 4:
            # Inverse exponential norm
            sigma = np.median(np.abs(d() - np.median(d())))
            d = -ch.exp(-d**2 / (2*sigma**2)) * sigma

        err = d.sum()

        if return_sigma:
            return sigma
        else:
            return err

    def energy_refinement(x,p0,p1,p2,Mcor,y0,y1,y2,errortype,do_print=False,sigmas_precomputed=None,return_sigmas=False):
        """Function to optimize.
        p0,p1,p2: Full points in all frames (uncorrected)
        Mcor: Correction matrix
        y0,y1,y2 : Cornerpoints defining the initial homographies

        x = [delta_y2_3_0, delta_y2_3_1, [delta_p2_0,delta_p2_1...]]
        """

        x_ch = ch.array(x)
        p0_ch = ch.array(p0)
        p1_ch = ch.array(p1)
        p2_ch = ch.array(p2)
        Mcor_ch = ch.array(Mcor)
        y0_ch = ch.array(y0)
        y1_ch = ch.array(y1)
        y2_ch = ch.array(y2)

        N = y1_ch.shape[0]

        y0_delta = x_ch[:8].reshape((4,2))
        y0_ch_new = y0_ch + y0_delta

        y2_delta = x_ch[8:].reshape((4,2))
        y2_ch_new = y2_ch + y2_delta
        
        # Estimate new H from y2_ch_new to y1_ch
        H_fwd = chumpy_get_H(y2_ch_new,y1_ch)
        H_bwd = chumpy_get_H(y0_ch_new,y1_ch)
        
        # Estimate homography from y0_ch_new to y2_ch_new
        H_combined = chumpy_get_H(y0_ch_new, y2_ch_new)

        if sigmas_precomputed is not None:
            sigma_fwd,sigma_bwd,sigma_combined = sigmas_precomputed
        else:
            sigma_fwd=sigma_bwd=sigma_combined=None
        
        # Note that H_fwd maps p2 to p1, and not p1 to p2!
        d_fwd = chumpy_compute_error(p1_ch,p2_ch,H_fwd,Mcor_ch,error=errortype,
                                     sigma_precomputed=sigma_fwd,
                                     return_sigma=return_sigmas)
        d_bwd = chumpy_compute_error(p1_ch,p0_ch,H_bwd,Mcor_ch,error=errortype,
                                     sigma_precomputed=sigma_bwd,
                                     return_sigma=return_sigmas)
        d_combined = chumpy_compute_error(p2_ch, p0_ch, H_combined, Mcor_ch,error=errortype,
                                          sigma_precomputed=sigma_combined,
                                          return_sigma=return_sigmas)

        if return_sigmas:
            # Don't actually compute the errors - instead, just return pre-computed sigmas.
            return d_fwd,d_bwd,d_combined
        else:
            err = d_fwd + d_bwd + d_combined
            derr = np.copy(err.dr_wrt(x_ch)).flatten()
            return err(),derr


    matches_all_frames = filter_features(matches_all_frames_, homographies_init)

    # -- Set up optimization --
    # Build fixed correction matrix (See MacLean)
    M = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    for x,y in matches_all_frames[:,:,0]:
        M += np.array([[1.0,0.0,-x],[0.0,1.0,-y],[-x,-y,x**2 + y**2]])
    Mcor = sqrtm(np.linalg.inv(M))

    print('--- Mcor ---')
    print(Mcor)
    print('---')

    def get_y2(shape,H):
        """
        Function to compute warped corner points
        """
        y1 = np.array([[0.0,0.0],
                      [shape[1],0.0],
                      [0.0,shape[0]],
                      [shape[1],shape[0]]])
        y1_ = np.c_[y1,np.ones(4)].T
        y2_ = H.dot(y1_)
        y2 = (y2_[:2,:] / y2_[2,:]).T
        return y2

    y0 = get_y2(shape,homographies_init[0])
    y1 = get_y2(shape,np.eye(3))
    y2 = get_y2(shape,homographies_init[1])

    # First iteration: square
    x0 = np.zeros(16)

    # The second to last argument is the errornorm.
    # 0 = square error
    # 1 = charbonnier
    # 2 = geman mcclure
    # 3 = corrected geman-mcclure
    args = [matches_all_frames[:,:,1],
            matches_all_frames[:,:,0],
            matches_all_frames[:,:,2],
            Mcor,
            y0,y1,y2,
            error_type,
            False]

    print('(MM) Initialization: Refining homographies...')
    t0 = time.time()
    res = optimize.minimize(energy_refinement,x0=x0,args=tuple(args),jac=True,method='L-BFGS-B',options={'disp': False, 'maxiter': 50})
    t1 = time.time()

    print('(MM) Initialization: Refinement took {} seconds'.format(t1-t0))


    delta_h_bwd = res.x[:8].reshape((4,2))
    delta_h_fwd = res.x[8:].reshape((4,2))

    homographies_refined = [nt.find_homography_normalized(y1,y0+delta_h_bwd),
                    nt.find_homography_normalized(y1,y2+delta_h_fwd)]

    if not (check_homography(homographies_refined[0], shape) and check_homography(homographies_refined[1],shape)):
        return homographies_init
    else:
        return homographies_refined


def compute_foes_svd_irls(matches_all_frames_, homographies_refined):
    """
    Compute initial set of FoEs.
    """

    matches_all_frames = filter_features(matches_all_frames_, homographies_refined)
    
    def estimate_foe_svd_norm_irls(points1, points2, init=None):
        """
        Estimate focus of expansion using least squares
        """
        A,b = points2system(points1[:,0],points1[:,1],points2[:,0],points2[:,1],norm=False)
        T = np.c_[A,-b]

        weights = np.ones((T.shape[0],1))

        mad = lambda x : 1.48 * np.median(np.abs(x - np.median(x)))

        for it in range(100):
            Tw = T * np.sqrt(weights)
            # Compute correction matrix
            M = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
            for i,p1 in enumerate(points1):
                M += weights[i]**2 * np.array([[1.0,0.0,-p1[0]],[0.0,1.0,-p1[1]],[-p1[0],-p1[1],p1[0]**2 + p1[1]**2]])

            Mcor = sqrtm(np.linalg.inv(M))

            X = Mcor.dot(Tw.T.dot(Tw)).dot(Mcor)
            U,S,V = np.linalg.svd(X)

            err = T.dot(Mcor).dot(V[-1,:])
            sigma = mad(err)
            weights[:,0] = 1.0 / (1 + (err/sigma)**2)

            weights /= weights.sum()

        ep1 = V[2,:]
        ep1 = Mcor.dot(ep1)
        ep1 = ep1[:2] / ep1[2]
        return ep1

    epipoles = []
    n_frames = matches_all_frames.shape[2]
    for i in range(1,n_frames):
        pt1 = matches_all_frames[:,:,0]
        pt2 = matches_all_frames[:,:,i]
        H = homographies_refined[i-1]
        pt2_H = remove_homography(pt2,H)
        epipoles.append(estimate_foe_svd_norm_irls(pt1,pt2_H))

    return epipoles



def compute_inliers(matches_all_frames, homographies_refined):
    inliers = np.zeros((matches_all_frames.shape[0],matches_all_frames.shape[2]-1))

    n_frames = matches_all_frames.shape[2]

    threshold = 1.0

    for i in range(1,n_frames):
        pt1 = matches_all_frames[:,:,0]
        pt2 = matches_all_frames[:,:,i]
        H = homographies_refined[i-1]
        pt2_H = remove_homography(pt2,H)

        inl = np.linalg.norm(pt1-pt2_H, axis=1) < threshold
        inliers[:,i-1] = inl
    
    return np.all(inliers>0,axis=1)


def remove_homography(p2,H):
    Hinv = np.linalg.inv(H)
    p2_ = np.c_[p2,np.ones(p2.shape[0])].T
    p3_ = Hinv.dot(p2_)
    p3 = (p3_[:2,:] / p3_[2,:]).T
    return p3


def points2system(x1,y1,x2,y2,norm=True):
    """
    Helper function to build a linear system from point matches.
    System can then be solved as Ax = b, with x being the epipole.

    """
    u = x2 - x1
    v = y2 - y1
    if norm:
        nrm = np.maximum(1e-3,np.sqrt(u**2 + v**2))
    else:
        nrm = 1.0
    un = u / nrm
    vn = v / nrm

    A = np.c_[-vn, un]
    b = -vn * x2 + un * y2
    return A,b



def filter_features(matches_all_frames, homographies, threshold=0.5):
    """Remove small features.

    Remove all features from matches_all_frames that are smaller than the
    given threshold after removal of the homographies.
    
    Returns
    -------
    matches_all_frames_filtered : array_like
        Output array containing only the filtered features.
    """
    return matches_all_frames

    pt1 = matches_all_frames[:,:,0]
    n_feats, _, n_frames = matches_all_frames.shape

    features_length = np.zeros((n_feats, n_frames-1))

    for frame in range(1,n_frames):
        pt2 = matches_all_frames[:,:,frame]
        pt2_H = remove_homography(pt2, homographies[frame-1])

        features_length[:,frame-1] = np.linalg.norm(pt1-pt2_H, axis=1)

    valid_features = np.all(features_length > threshold, axis=1)
    print('Filtering features: {} of {} features are large enough.'.format(valid_features.sum(), valid_features.shape[0]))
    return matches_all_frames[valid_features, :, :]



            


#
# New code to compute homographies with a different error term.
# The error term here computes how well a homography is induced by a plane in the scene.
# See H&Z, p336
#


def compute_initial_homographies_via_F_explicit(matches_all_frames, ransacReprojThreshold=1.0):
    """ Compute a set of initial homographies and the resulting intersecting points
        from the matches given in matches_all_frames
    """
    def estimate_model(X):
        submodels = []
        for frame in range(1,X.shape[2]):
            pt1 = X[:,:2,0]
            pt2 = X[:,:2,frame]
            if pt1.shape[0] < 6:
                #print('Too few points')
                return None
            else:
                F = cv2.findFundamentalMat(pt1.astype('float32'), pt2.astype('float32'), method=cv2.FM_8POINT)[0]
                submodels.append(F.ravel().astype('float64'))

        model = np.array([S.ravel() for S in submodels])
        # print('Model shape: {}'.format(model.shape))
        return model

    def estimate_inliers(X, model):
        dists = np.zeros((X.shape[0],X.shape[2]-1))
        for frame in range(X.shape[2]-1):
            F = model[frame,:].reshape((3,3)).astype('float64')
            
            pt1 = X[:,:,0]
            pt2 = X[:,:,frame+1]

            err = np.abs((pt2.T * F.dot(pt1.T)).sum(axis=0))

            dists[:,frame] = err

        inliers = np.all(dists<=ransacReprojThreshold,axis=1)

        return inliers


    matches_all_frames_ = np.concatenate((
        matches_all_frames,
        np.ones((matches_all_frames.shape[0],1,matches_all_frames.shape[2]))),
        axis=1)

    print(matches_all_frames.shape)

    LMEDS = False
    if LMEDS:
        homographies_model, inliers_ = lmeds.estimate(matches_all_frames_,
                                                    8,
                                                    estimate_model,
                                                    estimate_inliers,
                                                    do_print=True,
                                                    recompute_model=True,
                                                    seed=12345)
    else:
        homographies_model, inliers_ = ransac.estimate(matches_all_frames_,
                                                    8,
                                                    estimate_model,
                                                    estimate_inliers,
                                                    p_outlier=0.8,
                                                    do_print=True,
                                                    recompute_model=True,
                                                    seed=12345)



    if homographies_model is None:
        print('(EE) **** Could not compute a homography')
        raise Exception('NoHomographyFound')
    else:
        print('(MM) Number of inliers: {}'.format(inliers_.sum()))

    homographies_init = []

    # Generate plane points by finding the points that span the best homography
    matches_inliers = matches_all_frames_[inliers_, :, :]
    points_for_H = compute_points_ransac(matches_inliers, homographies_model, ransacReprojThreshold)

    print('(MM) Points to use: ')
    print(points_for_H[:,:,0])


    for i,f in enumerate(homographies_model):
        F = f.reshape((3,3))

        # Compute epipole from F
        U,S,V = np.linalg.svd(F)

        ep_ref = V[:,2]
        ep_2 = U[:,2]

        ex = np.array([[0, -ep_2[2], ep_2[1]],
                    [ep_2[2], 0, -ep_2[0]],
                    [-ep_2[1], ep_2[0], 0]])

        A = ex.dot(F)

        pt1_ = points_for_H[:,:,0]
        pt2_ = points_for_H[:,:,i+1]

        pt10 = pt1_[0,:]
        pt11 = pt1_[1,:]
        pt12 = pt1_[2,:]
        pt20 = pt2_[0,:]
        pt21 = pt2_[1,:]
        pt22 = pt2_[2,:]

        M = pt1_
        c0 = np.cross(pt20, ep_2)
        c1 = np.cross(pt21, ep_2)
        c2 = np.cross(pt22, ep_2)
        b0 = np.cross(pt20, A.dot(pt10)).T.dot(c0) / (np.linalg.norm(c0)**2)
        b1 = np.cross(pt21, A.dot(pt11)).T.dot(c1) / (np.linalg.norm(c1)**2)
        b2 = np.cross(pt22, A.dot(pt12)).T.dot(c2) / (np.linalg.norm(c2)**2)
        b = np.array([b0,b1,b2])

        sub = ep_2.reshape((3,1)) * np.linalg.inv(M).dot(b).reshape((1,3))
        H = A - sub
        H /= H[2,2]

        if not check_homography(H, [436,1024]):
            print('(EE) **** Could not compute a homography')
            raise Exception('NoHomographyFound')

        homographies_init.append(H)

        print('Initial EP {}: {} '.format(i, ep_ref[:2]/ep_ref[2]))
        print('Initial EP 2 {}: {}'.format(i, ep_2[:2] / ep_2[2]))
        print('Initial H {}'.format(i))
        print(H)

    return homographies_init



def compute_points_ransac(matches, fundamental_matrices, ransacReprojThreshold=1.0):
    """ Compute sampling points so that the resulting homography spans as many points
    as possible.

    """
    def estimate_model(X):
        """ Similar to above, the 'model' in this case are just the points."""
        # Directional vectors
        v1 = np.array([X[1,0,0] - X[0,0,0], X[1,1,0] - X[0,1,0]])
        v2 = np.array([X[2,0,0] - X[0,0,0], X[2,1,0] - X[0,1,0]])
        v3 = np.array([X[2,0,0] - X[1,0,0], X[2,1,0] - X[1,1,0]])

        n_v1 = np.linalg.norm(v1)
        n_v2 = np.linalg.norm(v2)
        n_v3 = np.linalg.norm(v3)

        # print([n_v1, n_v2, n_v3])

        if (n_v1 < 50) or (n_v2 < 50) or (n_v3 < 50):
            return None

        cos_deg = np.abs(v1[0]*v2[0]+v1[1]*v2[1]) / (n_v1*n_v2)
        # print(cos_deg)
        if cos_deg > 0.7:
            return None

        return X

    def estimate_inliers(X, model):
        """ Standard method to estimate number of inliers according to homography.
        """

        dists = np.zeros((X.shape[0],X.shape[2]-1))

        for frame,f in enumerate(fundamental_matrices):
            F = f.reshape((3,3))

            # Compute epipole from F
            U,S,V = np.linalg.svd(F)

            ep_ref = V[:,2]
            ep_2 = U[:,2]
            # ep_2 = ep_ref

            ex = np.array([[0, -ep_2[2], ep_2[1]],
                        [ep_2[2], 0, -ep_2[0]],
                        [-ep_2[1], ep_2[0], 0]])

            A = ex.dot(F)

            # pt1_ = matches_all_frames_[plane_use, :, 0]
            # pt2_ = matches_all_frames_[plane_use, :, i+1]
            pt1_ = model[:,:,0]
            pt2_ = model[:,:,frame+1]

            pt10 = pt1_[0,:]
            pt11 = pt1_[1,:]
            pt12 = pt1_[2,:]
            pt20 = pt2_[0,:]
            pt21 = pt2_[1,:]
            pt22 = pt2_[2,:]

            M = pt1_
            c0 = np.cross(pt20, ep_2)
            c1 = np.cross(pt21, ep_2)
            c2 = np.cross(pt22, ep_2)
            b0 = np.cross(pt20, A.dot(pt10)).T.dot(c0) / (np.linalg.norm(c0)**2)
            b1 = np.cross(pt21, A.dot(pt11)).T.dot(c1) / (np.linalg.norm(c1)**2)
            b2 = np.cross(pt22, A.dot(pt12)).T.dot(c2) / (np.linalg.norm(c2)**2)
            b = np.array([b0,b1,b2])

            sub = ep_2.reshape((3,1)) * np.linalg.inv(M).dot(b).reshape((1,3))
            H = A - sub
            H /= H[2,2]

            if not check_homography(H, [436,1024]):
                return np.zeros(dists.shape[0])==1

            # Compute number of inliers according to this H matrix

            Hinv = np.linalg.inv(H).astype('float32')
            H = H.astype('float32')
            pt1 = X[:,:2,0].astype('float32')
            pt2 = X[:,:2,frame+1].astype('float32')
            pt1_projected = cv2.perspectiveTransform(pt1.reshape((-1,1,2)),H).squeeze()
            pt2_projected = cv2.perspectiveTransform(pt2.reshape((-1,1,2)),Hinv).squeeze()
            errors1 = np.linalg.norm(pt2 - pt1_projected,axis=1)
            errors2 = np.linalg.norm(pt1 - pt2_projected,axis=1)
            dists[:,frame] = (errors1+errors2)/2.0
        inliers = np.all(dists<=ransacReprojThreshold,axis=1)
        return inliers



    points, inliers = ransac.estimate(matches,
                                      3,
                                      estimate_model,
                                      estimate_inliers,
                                      p_outlier=0.8,
                                      do_print=True,
                                      recompute_model=False,
                                      min_iters=10000,
                                      seed=12345)

    print('(MM) {} of {} points are inliers according to H.'.format(inliers.sum(), np.size(inliers)))
    return points



