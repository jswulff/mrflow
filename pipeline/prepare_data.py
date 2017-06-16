#! /usr/bin/env python2

import numpy as np
import sys,os,time
import argparse

import cv2
from skimage import img_as_float
from skimage import morphology


cpath = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(cpath)
if 'MRFLOW_HOME' in os.environ.keys():
    sys.path.append(os.environ['MRFLOW_HOME'])

# Local imports
import parameters
from utils import flow_io as fio
from utils import plot_debug as pd
from utils import print_exception as pe
from utils import epicwrapper

from rigidity import occlusions


def main():
    """
    Load images, flow, rigidity maps.
    Read in parameters.
    """
    print('Starting data preparation...')

    parser = argparse.ArgumentParser()
    parser.add_argument('image_prev', type=str, help='Previous image')
    parser.add_argument('image_current', type=str, help='Current image')
    parser.add_argument('image_next', type=str, help='Next image')
    parser.add_argument('--flow_fwd', type=str, default='', help='Forward flow field')
    parser.add_argument('--flow_bwd', type=str, default='', help='Backward flow field')
    parser.add_argument('--backflow_fwd', type=str, default='', help='Forward back flow field')
    parser.add_argument('--backflow_bwd', type=str, default='', help='Backward back flow field')
    parser.add_argument('--rigidity', type=str, default='', help='Initial rigidity')
    parser.add_argument('--tempdir', type=str, default='.', help='Directory to store data files')
    parser.add_argument('--flow_fwd_gt', type=str, default='', help='Ground truth forward flow')
    parser.add_argument('--rigidity_gt', type=str, default='', help='Ground truth rigidity')

    # Add custom parameters to parser
    parameters.add_parameters_argparse(parser)

    # Parse parameters
    args = parser.parse_args()

    print('=== ARGUMENTS ===')
    for k in sorted(vars(args).keys()):
        v = vars(args)[k]
        print('\t {} \t : \t {}'.format(k,v))
    print('=================')


    # If forward flow is given but not found, exit.
    if args.flow_fwd and (not os.path.isfile(args.flow_fwd)):
        # File does not exists.
        print('(EE)')
        print('(EE) Forward initial flow given but not found. Quitting.')
        print('(EE)')
        sys.exit(1)

    # Check if at least the current and next image exist.
    if not (os.path.isfile(args.image_current) and os.path.isfile(args.image_next)):
        print('(EE)')
        print('(EE) Current and next frames not found. Quitting.')
        print('(EE)')
        sys.exit(1)

    try:
        # Load images
        images = [img_as_float(cv2.imread(p)[:,:,::-1]) for p in [args.image_prev, args.image_current, args.image_next]]

        # Flow 
        EPIC_PRESET = 'sintel'
        if args.flow_fwd:
            flow_fwd = fio.flow_read(args.flow_fwd)
        else:
            flow_fwd = epicwrapper.compute_epicflow(images[1],images[2], preset=EPIC_PRESET)
        if args.flow_bwd:
            flow_bwd = fio.flow_read(args.flow_bwd)
        else:
            flow_bwd = epicwrapper.compute_epicflow(images[1],images[0],preset=EPIC_PRESET)
        flow = [flow_bwd, flow_fwd]

        # GT Flow
        if not args.flow_fwd_gt == '':
            if args.flow_fwd_gt.endswith('flo'):
                # Read Sintel-style .flo file.
                flow_fwd_gt = fio.flow_read(args.flow_fwd_gt)
                flow_fwd_gt_valid = np.ones(flow_fwd_gt[0].shape,dtype='bool')
            elif args.flow_fwd_gt.endswith('png'):
                ugt,vgt,flow_fwd_gt_valid = fio.flow_read_png(args.flow_fwd_gt)
                flow_fwd_gt = [ugt,vgt]
            else:
                flow_fwd_gt = None
                flow_fwd_gt_valid = None
        else:
            flow_fwd_gt = None
            flow_fwd_gt_valid = None

        # Rigidity
        if args.rigidity:
            rigidity = img_as_float(cv2.imread(args.rigidity))[:,:,0]
        else:
            # 0.8 is the general prior for rigidity
            rigidity = 0.8 * np.ones(images[1].shape[:2])
        rigidity_thresholded = morphology.binary_erosion(rigidity > 0.5, np.ones((3,3)))

        if not args.rigidity_gt == '':
            rigidity_gt = img_as_float(cv2.imread(args.rigidity_gt))[:,:,0] > 0.5
        else:
            rigidity_gt = None



        #
        # Compute initial occlusions
        #
        if args.backflow_fwd and args.backflow_bwd:
            backflow_fwd = fio.flow_read(args.backflow_fwd)
            backflow_bwd = fio.flow_read(args.backflow_bwd)
            # No need to store the backflow, since we only use it to compute the occlusion.
            occ_bwd, occ_fwd, occ_both = occlusions.computeOcclusionsFromConsistency(
                flow, [backflow_bwd, backflow_fwd], threshold=args.occlusion_threshold)
            occ = [occ_bwd, occ_fwd]

        else:
            occ = occlusions.compute_occlusions_from_flow(images, flow)
            occ_both = np.zeros_like(occ[0]) > 0


        if args.debug_save_frames:
            pd.plot_image(rigidity, '[01 Data input] Rigidity', outpath=args.tempdir)
            pd.plot_image(rigidity_thresholded, '[01 Data input] Rigidity, thresholded', outpath=args.tempdir)
            pd.plot_image(occ[0] + occ[1]*2 + occ_both*4, '[01 Data input] Computed occlusions', vmin=0, vmax=4, outpath=args.tempdir)
            pd.plot_flow(flow_bwd[0],flow_bwd[1],'[01 Data input] Initial flow 1-0', outpath=args.tempdir)
            pd.plot_flow(flow_fwd[0],flow_fwd[1],'[01 Data input] Initial flow 1-2', outpath=args.tempdir)
            if args.backflow_fwd:
                pd.plot_flow(backflow_fwd[0], backflow_fwd[1], '[01 Data input] Initial flow 2-1', outpath=args.tempdir)
            if args.backflow_bwd:
                pd.plot_flow(backflow_bwd[0], backflow_bwd[1], '[01 Data input] Initial flow 0-1', outpath=args.tempdir)


        params = args

        data_out = {'images': images,
                'flow': flow,
                'rigidity': rigidity,
                'rigidity_thresholded': rigidity_thresholded,
                'flow_fwd_gt': flow_fwd_gt,
                'flow_fwd_gt_valid': flow_fwd_gt_valid,
                'rigidity_gt': rigidity_gt,
                'occlusions': occ,
                'debug_tempdir': args.tempdir,
                'params': params}

        # Check if > 50% of init pixels are given as non-rigid. In that case, just
        # return initial flow. (If a flag ERROR_OVERRIDE is present in the
        # temporary data, each sub-program just relays the input data to the next
        # sub-program.)
        print(rigidity.shape)

        #if rigidity_thresholded.sum() < (0.25 * np.size(rigidity_thresholded)):
            #print('(EE) **** More than 50 percent initialized as non-rigid. Returning initial flow.')
            #data_out['error_override'] = True
        #else:
            #data_out['error_override'] = False


        # Check if too few pixels are never occluded (we want at least 50%)
        not_occluded = (occ[0]==0)*(occ[1]==0)
        if not_occluded.sum() < 0.5 * not_occluded.size:
            print('(EE) Too many pixels are occluded!')
            data_out['error_override'] = True
        else:
            data_out['error_override'] = False
    
    except:
        pe.print_exception()
        if args.flow_fwd:
            # The flow exists, but some other data was not found, most likely.
            flow = [None, fio.flow_read(args.flow_fwd)]
        else:
            # We checked already that the current and next frame exists.
            flow_fwd = epicwrapper.compute_epicflow(
                    args.image_current,
                    args.image_next)
            flow = [None, flow_fwd]

        data_out = {'flow': flow, 'error_override': True}



    # Save all
    print('Data preparation: Saving temp file...')
    path_out = os.path.join(args.tempdir, '01_data_init.npz')
    np.savez(path_out, **data_out)

    

if __name__ == '__main__':
    main()








