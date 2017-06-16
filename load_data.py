#! /usr/bin/env python2
import sys,os

import numpy as np

MRFLOW_HOME = os.environ['MRFLOW_HOME']
sys.path.append(MRFLOW_HOME)

from utils.imgutils import img_as_float as img_as_float
from utils import print_exception as pe
from utils import compute_figure
from utils import flow_io as fio
from utils import plot_debug as pd
try:
    from utils import epicwrapper
except:
    epicwrapper = None

from rigidity import occlusions

import cv2


def load_initial_data(
        path_image_prev, path_image_current, path_image_next,
        path_flow_fwd='', path_flow_bwd='',
        path_backflow_fwd='', path_backflow_bwd='',
        path_rigidity='',
        path_flow_fwd_gt='',
        path_rigidity_gt='',
        params=None):
    """Load initial data, returning data_in structure.
    """

    # If forward flow is given but not found, exit.
    if path_flow_fwd and (not os.path.isfile(path_flow_fwd)):
        # File does not exists.
        print('(EE)')
        print('(EE) Forward initial flow given as parameter, but file not found. Quitting.')
        print('(EE)')
        sys.exit(1)

    # Check if at least the current and next image exist.
    if not (os.path.isfile(path_image_current) and os.path.isfile(path_image_next)):
        print('(EE)')
        print('(EE) Current and next frames not found. Quitting.')
        print('(EE)')
        sys.exit(1)

    try:
        # Load images
        images = [img_as_float(cv2.imread(p)[:,:,::-1]) for p in [path_image_prev, path_image_current, path_image_next]]

        # Flow
        # We either pre-load all flow fields, or compute all flow fields.
        if path_flow_fwd and path_flow_bwd and path_backflow_fwd and path_backflow_bwd:
            flow_fwd = fio.flow_read(path_flow_fwd)
            flow_bwd = fio.flow_read(path_flow_bwd)
            backflow_fwd = fio.flow_read(path_backflow_fwd)
            backflow_bwd = fio.flow_read(path_backflow_bwd)
        else:
            # Check if we have pre-computed the flow fields already
            recompute_flow = False
            for initflowname in ['init_flow_fwd.flo',
                                 'init_flow_bwd.flo',
                                 'init_backflow_fwd.flo',
                                 'init_backflow_bwd.flo']:
                if not os.path.isfile(os.path.join(params.tempdir, initflowname)):
                    recompute_flow = True

            if not recompute_flow:
                flow_fwd = fio.flow_read(os.path.join(params.tempdir,'init_flow_fwd.flo'))
                flow_bwd = fio.flow_read(os.path.join(params.tempdir,'init_flow_bwd.flo'))
                backflow_fwd = fio.flow_read(os.path.join(params.tempdir,'init_backflow_fwd.flo'))
                backflow_bwd = fio.flow_read(os.path.join(params.tempdir,'init_backflow_bwd.flo'))

            else:
                EPIC=False
                print('Computing flow fields... ')
                if EPIC:
                    # Compute optical flow using EPIC flow
                    EPIC_PRESET='kitti'
                    flow_fwd = epicwrapper.compute_epicflow(images[1],images[2], preset=EPIC_PRESET)
                    flow_bwd = epicwrapper.compute_epicflow(images[1],images[0], preset=EPIC_PRESET)
                    backflow_fwd = epicwrapper.compute_epicflow(images[2],images[1], preset=EPIC_PRESET)
                    backflow_bwd = epicwrapper.compute_epicflow(images[0],images[1], preset=EPIC_PRESET)
                    fio.flow_write(params.tempdir + '/init_flow_fwd.flo', flow_fwd)
                    fio.flow_write(params.tempdir + '/init_flow_bwd.flo', flow_bwd)
                    fio.flow_write(params.tempdir + '/init_backflow_fwd.flo', backflow_fwd)
                    fio.flow_write(params.tempdir + '/init_backflow_bwd.flo', backflow_bwd)
                else:
                    # Compute optical flow using DeepFlow (included in OpenCV)
                    D = cv2.optflow.createOptFlow_DeepFlow()
                    cvt = lambda x : (cv2.cvtColor(x.astype('float32'),cv2.COLOR_RGB2GRAY)*255).astype('uint8')
                    I0_bw = cvt(images[0])
                    I1_bw = cvt(images[1])
                    I2_bw = cvt(images[2])
                    h,w = I0_bw.shape
                    flow_temp = np.zeros((h,w,2))

                    print('\tfwd')
                    flow_fwd = D.calc(I1_bw,I2_bw,flow_temp)
                    print('\tbwd')
                    flow_bwd = D.calc(I1_bw,I0_bw,flow_temp)
                    print('\tbackflow, fwd')
                    backflow_fwd = D.calc(I2_bw,I1_bw,flow_temp)
                    print('\tbackflow, bwd')
                    backflow_bwd = D.calc(I0_bw,I1_bw,flow_temp)

                    flow_fwd = [flow_fwd[:,:,0],flow_fwd[:,:,1]]
                    flow_bwd = [flow_bwd[:,:,0],flow_bwd[:,:,1]]
                    backflow_fwd = [backflow_fwd[:,:,0],backflow_fwd[:,:,1]]
                    backflow_bwd = [backflow_bwd[:,:,0],backflow_bwd[:,:,1]]


        flow = [flow_bwd, flow_fwd]

        # GT Flow
        if not path_flow_fwd_gt == '':
            if path_flow_fwd_gt.endswith('flo'):
                # Read Sintel-style .flo file.
                flow_fwd_gt = fio.flow_read(path_flow_fwd_gt)
                flow_fwd_gt_valid = np.ones(flow_fwd_gt[0].shape,dtype='bool')
            elif path_flow_fwd_gt.endswith('png'):
                # Read KITTI-style .png flow file
                ugt,vgt,flow_fwd_gt_valid = fio.flow_read_png(path_flow_fwd_gt)
                flow_fwd_gt = [ugt,vgt]
            else:
                flow_fwd_gt = None
                flow_fwd_gt_valid = None
        else:
            flow_fwd_gt = None
            flow_fwd_gt_valid = None

        # Rigidity
        if path_rigidity:
            rigidity = img_as_float(cv2.imread(path_rigidity))[:,:,0]
        else:
            # 0.6 is the general prior for rigidity
            rigidity = 0.6 * np.ones(images[1].shape[:2])

        rigidity_thresholded = cv2.erode((rigidity>0.5).astype('uint8'), np.ones((3,3),np.uint8)) > 0

        if not path_rigidity_gt == '':
            rigidity_gt = img_as_float(cv2.imread(path_rigidity_gt))[:,:,0] > 0.5
        else:
            rigidity_gt = None



        #
        # Compute initial occlusions
        #
        if params.occlusion_reasoning:
            # No need to store the backflow, since we only use it to compute the occlusion.
            occ_bwd, occ_fwd, occ_both = occlusions.computeOcclusionsFromConsistency(
                flow, [backflow_bwd, backflow_fwd], threshold=params.occlusion_threshold)
        else:
            occ_bwd = np.zeros(images[1].shape[:2])>0
            occ_fwd = np.zeros(images[1].shape[:2])>0
            occ_both = np.zeros_like(occ_bwd)>0

        occ = [occ_bwd, occ_fwd]


        if params.debug_save_frames:
            pd.plot_image(rigidity, '[01 Data input] Rigidity', outpath=params.tempdir)
            pd.plot_image(rigidity_thresholded, '[01 Data input] Rigidity, thresholded', outpath=params.tempdir)
            pd.plot_image(occ[0] + occ[1]*2 + occ_both*4, '[01 Data input] Computed occlusions', vmin=0, vmax=4, outpath=params.tempdir)
            pd.plot_flow(flow_bwd[0],flow_bwd[1],'[01 Data input] Initial flow 1-0', outpath=params.tempdir)
            pd.plot_flow(flow_fwd[0],flow_fwd[1],'[01 Data input] Initial flow 1-2', outpath=params.tempdir)
            if path_backflow_fwd and params.occlusion_reasoning:
                pd.plot_flow(backflow_fwd[0], backflow_fwd[1], '[01 Data input] Initial flow 2-1', outpath=params.tempdir)
            if path_backflow_bwd and params.occlusion_reasoning:
                pd.plot_flow(backflow_bwd[0], backflow_bwd[1], '[01 Data input] Initial flow 0-1', outpath=params.tempdir)




        data_out = {'images': images,
                'flow': flow,
                'rigidity': rigidity,
                'rigidity_thresholded': rigidity_thresholded,
                'flow_fwd_gt': flow_fwd_gt,
                'flow_fwd_gt_valid': flow_fwd_gt_valid,
                'rigidity_gt': rigidity_gt,
                'occlusions': occ,
                'debug_tempdir': params.tempdir,
                'params': params}

        # Check if too few pixels are never occluded (we want at least 50%)
        not_occluded = (occ[0]==0)*(occ[1]==0)
        if not_occluded.sum() < 0.3 * not_occluded.size:
            print('(EE) Too many pixels are occluded!')
            data_out['error_override'] = True
        else:
            data_out['error_override'] = False

    except:
        print('(EE) Exception in load_data.py')
        pe.print_exception()
        if path_flow_fwd:
            # The flow exists, but some other data was not found, most likely.
            flow = [None, fio.flow_read(path_flow_fwd)]
        else:
            # We checked already that the current and next frame exists.
            flow_fwd = epicwrapper.compute_epicflow(
                    path_image_current,
                    path_image_next)
            flow = [None, flow_fwd]

        data_out = {'flow': flow, 'error_override': True}

    return data_out

