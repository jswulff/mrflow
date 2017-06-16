#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

sys.path.append(os.environ['MRFLOW_HOME'])
# Local imports

from utils import plot_debug as pd
from utils import print_exception as pe
from utils import flow_io as fio
from utils import flow_viz as fviz



def get_error_images(u1,v1,u2,v2,ugt,vgt,thresh=3.0, valid=None):
    if valid is None:
        valid = np.ones_like(u1)>0

    error1 = np.sqrt((u1-ugt)**2 + (v1-vgt)**2)
    error2 = np.sqrt((u2-ugt)**2 + (v2-vgt)**2)

    inds1 = valid * (error2<error1)
    inds2 = valid * (error1<error2)

    img_better = np.zeros((u1.shape[0],u1.shape[1],3))
    img_better[:,:,0][inds1] = np.minimum(error1[inds1]-error2[inds1],1.0)
    img_better[:,:,1][inds2] = np.minimum(error2[inds2]-error1[inds2],1.0)

    img_error_thresh = np.zeros((u1.shape[0],u1.shape[1],3))
    img_error_thresh[:,:,0][valid] = error1[valid] >= thresh
    img_error_thresh[:,:,1][valid] = error2[valid] >= thresh

    return img_better, img_error_thresh




def compute_errors(ugt,vgt,uest,vest,rigidity,valid=None):
    if valid is None:
        valid = np.ones_like(rigidity)

    valid = valid > 0
    error = np.sqrt((ugt-uest)**2 + (vgt-vest)**2)

    epe_rigid = error[valid * (rigidity>0)].mean()
    epe_all = error[valid].mean()

    perc_rigid = (error[valid * (rigidity>0)]>3).sum()  / float(error[valid * (rigidity>0)].size)
    perc_all = (error[valid]>3).sum() / float(error[valid].size)

    if (rigidity==0).sum() < 2:
        epe_nonrigid = 0
        perc_nonrigid = 0
    else:
        epe_nonrigid = error[valid * (rigidity==0)].mean()
        perc_nonrigid = (error[valid * (rigidity==0)]>3).sum() / float(error[valid * (rigidity==0)].size)

    return epe_rigid, epe_nonrigid, epe_all, perc_rigid, perc_nonrigid, perc_all



def save_frames(u,v,uinit,vinit,ugt,vgt,flow_fwd_gt_valid,rigidity,params):
    Iuv_gt = fviz.computeFlowImage(ugt,vgt)
    Iuv_est = fviz.computeFlowImage(u,v)
    Iuv_init = fviz.computeFlowImage(uinit,vinit)

    I_error,I_thresh = get_error_images(u, v, uinit, vinit, ugt, vgt, valid=flow_fwd_gt_valid)

    error_est =np.sqrt((u-ugt)**2 + (v-vgt)**2)
    error_est[flow_fwd_gt_valid==0] = 0
    error_init =np.sqrt((uinit-ugt)**2 + (vinit-vgt)**2)
    error_init[flow_fwd_gt_valid==0] = 0

    err_max = max(error_est[rigidity>0].max(), error_init[rigidity>0].max())

    pd.plot_image(Iuv_gt, '[07 Flow evaluation] Ground truth flow', outpath=params.tempdir)
    pd.plot_image(Iuv_est, '[07 Flow evaluation] Estimated flow.', outpath=params.tempdir)
    pd.plot_image(Iuv_init, '[07 Flow evaluation] Initial flow.', outpath=params.tempdir)
    pd.plot_image(error_est, '[07 Flow evaluation] Error image estimated flow.',vmin=0,vmax=err_max, outpath=params.tempdir)
    pd.plot_image(error_init, '[07 Flow evaluation] Error image initial flow.',vmin=0,vmax=err_max, outpath=params.tempdir)
    pd.plot_image(I_error, '[07 Flow evaluation] Comparison: Absolute', outpath=params.tempdir)
    pd.plot_image(I_thresh, '[07 Flow evaluation] Comparison: Threshold', outpath=params.tempdir)




