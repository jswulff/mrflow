import numpy as np
import cv2

def get_residual_flow(x,y,u,v,H):
    """ Removes homography from flow

    Parameters: x,y,u,v,H.
    """
    x2 = x + u
    y2 = y + v
    pt2_ = np.c_[x2.ravel(),y2.ravel()].astype('float32')
    pt1_ = np.c_[x.ravel(),y.ravel()].astype('float32')
    
    Hinv = np.linalg.inv(H)
    uv_ = cv2.perspectiveTransform(pt2_[:,np.newaxis,:],Hinv).squeeze() - pt1_
    return uv_[:,0].reshape(x.shape), uv_[:,1].reshape(x.shape)



def get_full_flow(u_res,v_res,H,x=None,y=None,shape=None):
    """ Adds homography back into flow

    Parameters: x,y,u,v,H
    """

    if (shape is None) and (x is None or y is None):
        print('Please provide either x/y or the shape.')

    if x is None and y is None:
        x,y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))

    x2 = x + u_res
    y2 = y + v_res
    pt2_ = np.c_[x2.ravel(),y2.ravel()].astype('float32')
    pt1_ = np.c_[x.ravel(),y.ravel()].astype('float32')
    uv_ = cv2.perspectiveTransform(pt2_[:,np.newaxis,:],H).squeeze() - pt1_
    return uv_[:,0].reshape(x.shape), uv_[:,1].reshape(x.shape)



def homography2flow(x,y,H):
    """ Convert homography to flow field

    """
    pt1_ = np.c_[x.ravel(),y.ravel()].astype('float32')
    uv = cv2.perspectiveTransform(pt1_[:,np.newaxis,:],H).squeeze() - pt1_

    return uv[:,0].reshape(x.shape), uv[:,1].reshape(x.shape)


