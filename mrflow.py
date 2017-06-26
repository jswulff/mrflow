#! /usr/bin/env python2

import sys,os
import argparse
import time

import numpy as np

import cv2

MRFLOW_HOME = os.environ['MRFLOW_HOME']
sys.path.append(MRFLOW_HOME)

from utils import print_exception as pe
from utils import compute_figure
from utils import flow_io as fio
from utils import flow_viz as fviz
from utils import structure2image

import load_data
import parameters


from pipeline import match_features
from pipeline import initial_alignment
from pipeline import compute_structure
from pipeline import optimize_variational_refinement as optimize
from pipeline import structure2flow
from pipeline import evaluate_flow


def compute_mrflow(paths,params):
    """ Process a single frame.
    """

    tdir = params.tempdir

    print('(MM) PATHS:')
    for k in sorted(paths.keys()):
        print('(MM)\t {}: {}'.format(k,paths[k]))

    # Load data
    data_in = load_data.load_initial_data(params=params, **paths)

    override = False
    if 'error_override' in data_in.keys() and data_in['error_override'] is True:
        print('(EE) Overriding computation...')
        u,v = data_in['flow'][1]
        flow = data_in['flow']
        override=True
        rigidity_output = np.ones_like(u)>0
        structure_optimized = np.ones_like(u)



    else:
        try:
            # Load data from input structure
            flow = data_in['flow']
            #flow_fwd_gt = data_in['flow_fwd_gt'] if 'flow_fwd_gt' in data_in.keys() else None
            rigidity_thresholded = data_in['rigidity_thresholded']
            rigidity = data_in['rigidity']
            images = data_in['images']
            occlusions = data_in['occlusions']
            # params = data_in['params'].tolist()
        except:
            print('Exception occured in file {}'.format(fil))
            pe.print_exception()
            raise Exception()

        if params.debug_use_rigidity_gt:
            rigidity_thresholded = data_in['rigidity_gt'] > 0.5
            rigidity = data_in['rigidity_gt'] > 0.5

        try:
            # Step 2: Feature matching
            tmatch_0 = time.time()
            (matches_pairwise,
                    matches_all_frames,
                    outliers_large_pairwise,
                    outliers_F_pairwise) = match_features.generate_features(
                            images, flow, occlusions, rigidity_thresholded, params)


            tmatch_1 = time.time()
            tinit_0 = time.time()

            # Step 3: Alignment
            (homographies,
                    epipoles,
                    plane_global) = initial_alignment.align_frames(
                            images,
                            matches_all_frames,
                            matches_pairwise,
                            params)

            # Step 4: Building structure
            (structure,
             homographies_refined,
             mu,
             B,
             epipoles_new,
             rigidity_refined,
             occlusions) = compute_structure.compute_structure(
                 images,
                 flow,
                 rigidity,
                 occlusions,
                 homographies,
                 epipoles,
                 params)

            tinit_1 = time.time()
            topt_0 = time.time()

            # Step 5: Optimize structure
            (structure_optimized,) = optimize.optimize(
                    images,
                    structure,
                    rigidity_refined,
                    epipoles_new,
                    homographies_refined,
                    B,
                    mu,
                    occlusions,
                    params)
            topt_1 = time.time()
            tcomb_0 = time.time()

            # Step 6: Re-compute flow.
            rigidity_output, u,v = structure2flow.combine_flow(
                    structure_optimized, # A
                    flow[1], # flow_init
                    rigidity_refined,
                    epipoles_new,
                    mu,
                    homographies_refined,
                    B,
                    images,
                    params)

            tcomb_1 = time.time()

            print('=============== Timing information ====================')
            print('Feature matching: \t {0:2.3f} seconds'.format(tmatch_1-tmatch_0))
            print('Alignment & rigidity computation: \t {0:2.3f} seconds'.format(tinit_1-tinit_0))
            print('Optimization: \t {0:2.3f} seconds'.format(topt_1-topt_0))
            print('Merging: \t {0:2.3f} seconds'.format(tcomb_1-tcomb_0))
            print('=======================================================')



        except:
            # Some mistake occured.
            print('==== EXCEPTION ====')
            pe.print_exception()

            # Just use initial flow.
            u,v = flow[1]
            rigidity_output = np.ones_like(u)>0
            structure_optimized = np.ones_like(u)




    #
    # Write flow and images
    #

    # Save flow output
    fio.flow_write(params.tempdir + '/' + 'flow.flo', u, v)
    # Save structure output
    np.save(params.tempdir + '/' + 'structure.npy', structure_optimized)

    # fio.flow_write_png(u, v, params.tempdir + '/' + 'flow.png')
    Iest = fviz.computeFlowImage(u,v)
    uinit,vinit = flow[1]
    Iinit = fviz.computeFlowImage(uinit,vinit)

    if override:
        rigidity_output = np.ones_like(u)>0
    rigidity_viz = (np.dstack((rigidity_output,rigidity_output,rigidity_output))*255).astype('uint8')

    structure_viz = structure2image.structure2image(structure_optimized,
                                                    rigidity_output,
                                                    cmap='viridis')

    flowdiff = np.sqrt((u-uinit)**2 + (v-vinit)**2)
    flowdiff = 255.0 * flowdiff / max(1.0,flowdiff.max())
    flowdiff = np.dstack((flowdiff, flowdiff, flowdiff)).astype('uint8')
    Iviz = np.vstack(( np.hstack((Iinit, Iest)), np.hstack((flowdiff, rigidity_viz)) ))
    # Save visualization image
    def saveim(fname, I):
        if I.ndim == 3:
            cv2.imwrite(fname, I[:,:,::-1])
        else:
            cv2.imwrite(fname, I)

    saveim(params.tempdir + '/comparison.png', Iviz)
    saveim(params.tempdir + '/structure.png', structure_viz)
    saveim(params.tempdir + '/flow_viz.png', Iest)

    # Save rigidity output
    saveim(params.tempdir + '/rigidity.png', (rigidity_output>0).astype('uint8')*255)




    ## Compute errors
    if 'flow_fwd_gt' in data_in and data_in['flow_fwd_gt'] is not None:
        ugt,vgt = data_in['flow_fwd_gt']
        flow_gt_valid = data_in['flow_fwd_gt_valid']

        uinit,vinit = data_in['flow'][1]


        # Do we have a GT rigidity map?
        if 'rigidity_gt' in data_in.keys() and data_in['rigidity_gt'].tolist() is not None:
            rigidity_gt = data_in['rigidity_gt']
        else:
            rigidity_gt = np.ones_like(ugt)>0

        if 'flow_fwd_gt_valid' in data_in.keys() and data_in['flow_fwd_gt_valid'].tolist() is not None:
            flow_fwd_gt_valid = data_in['flow_fwd_gt_valid']
        else:
            flow_fwd_gt_valid = np.ones_like(ugt)>0


        # Compute error of estimated flow
        (epe_est_rigid,
         epe_est_nonrigid,
         epe_est_all,
         perc_est_rigid,
         perc_est_nonrigid,
         perc_est_all) = evaluate_flow.compute_errors(ugt,vgt,u,v,rigidity_gt,valid=flow_fwd_gt_valid)

        # Compute error of initial flow
        (epe_init_rigid,
         epe_init_nonrigid,
         epe_init_all,
         perc_init_rigid,
         perc_init_nonrigid,
         perc_init_all) = evaluate_flow.compute_errors(ugt,vgt,flow[1][0],flow[1][1],rigidity_gt,valid=flow_fwd_gt_valid)

        #
        # Display errors
        #

        print('******************** EPE STATS ********************')
        print('\t\t\tSTATIC  \tNON-STATIC\tALL')
        print('Initial flow:   \t{0:2.5f} \t{1:2.5f} \t{2:2.5f}'.format(epe_init_rigid,epe_init_nonrigid,epe_init_all))
        print('Estimated flow: \t{0:2.5f} \t{1:2.5f} \t{2:2.5f}'.format(epe_est_rigid,epe_est_nonrigid,epe_est_all))
        print('***************************************************')

        print('******************** PERC STATS ********************')
        print('\t\t\tSTATIC  \tNON-STATIC\tALL')
        print('Initial flow:   \t{0:2.5f} \t{1:2.5f} \t{2:2.5f}'.format(perc_init_rigid,perc_init_nonrigid,perc_init_all))
        print('Estimated flow: \t{0:2.5f} \t{1:2.5f} \t{2:2.5f}'.format(perc_est_rigid,perc_est_nonrigid,perc_est_all))
        print('***************************************************')



        if vars(params).has_key('debug_compute_figure') and vars(params)['debug_compute_figure'] == 1:
            # Compute teaser image
            compute_figure.plot_figure_1(images, rigidity_refined, structure_optimized, (u,v), (ugt,vgt))

        if vars(params).has_key('debug_compute_figure') and vars(params)['debug_compute_figure'] == 2:
            compute_figure.plot_figure_2(images, flow, rigidity, structure[1], occlusions, rigidity_refined, structure_optimized, (u,v))

        if vars(params).has_key('debug_compute_figure') and vars(params)['debug_compute_figure'] == 5:
            compute_figure.plot_figure_5(images, rigidity_refined, structure_optimized, (u,v), (uinit,vinit), (ugt,vgt), flow_gt_valid )

        if vars(params).has_key('debug_compute_figure') and vars(params)['debug_compute_figure'] == 6:
            # Compute supmat results figure
            compute_figure.plot_figure_6(images, rigidity_refined, structure_optimized, (u,v), (uinit,vinit), (ugt,vgt), flow_gt_valid)


        #
        # Output error images
        #
        if params.debug_save_frames:
            evaluate_flow.save_frames(
                u,
                v,
                flow[1][0],
                flow[1][1],
                ugt,
                vgt,
                flow_fwd_gt_valid>0,
                rigidity_output,
                params)


        #
        # Output CSV file
        #
        data_array = [ (epe_init_rigid, epe_est_rigid),
                       (epe_init_nonrigid, epe_est_nonrigid),
                       (epe_init_all, epe_est_all),
                       (perc_init_rigid, perc_est_rigid),
                       (perc_init_nonrigid, perc_est_nonrigid),
                       (perc_init_all, perc_est_all)]

        names = ['epe_rigid.csv',
                'epe_nonrigid.csv',
                'epe_all.csv',
                'perc_rigid.csv',
                'perc_nonrigid.csv',
                'perc_all.csv']

        for csv_data, csv_name in zip(data_array, names):
            fname = os.path.join(params.tempdir, csv_name)
            with open(fname, 'w') as fil:
                fil.write(',InitialFlow,MR-Flow\n')
                fil.write('{},{},{}\n'.format(params.tempdir, csv_data[0],csv_data[1]))




def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('frame0', type=str, help='Frame at T-1')
    parser.add_argument('frame1', type=str, help='Frame at T')
    parser.add_argument('frame2', type=str, help='Frame at T+1')
    # Optional arguments
    parser.add_argument('--rigidity', type=str, default='', help='Rigidity initialization')
    parser.add_argument('--flow_fwd', type=str, default='', help='Initial flow in forward direction (T to T+1)')
    parser.add_argument('--backflow_fwd', type=str, default='', help='Initial backflow in forward direction (T+1 to T)')
    parser.add_argument('--flow_bwd', type=str, default='', help='Initial flow in backward direction (T to T-1)')
    parser.add_argument('--backflow_bwd', type=str, default='', help='Initial backflow in backward direction (T-1 to T)')

    # Override initialization arguments
    parser.add_argument('--no_init', action='store_true', help='Do not provide initial rigidity and flow. Instead, the rigidity is set to constant, and the flow is computed using DiscreteFlow. Warning: This will severely impact performance!')

    # GT data, if an evaluation is desired
    parser.add_argument('--flow_fwd_gt', type=str, default='', help='GT flow in forward direction (used only for evaluation)')
    parser.add_argument('--rigidity_gt', type=str, default='', help='GT rigidity (used only for evaluation)')

    # Add algorithm parameters
    parameters.add_parameters_argparse(parser)

    # Parse all parameters
    args = parser.parse_args()

    if not args.rigidity:
        vars(args)['rigidity_weight_cnn'] = 0.0

    #
    # Display all arguments
    #
    print('=== ARGUMENTS ===')
    for k in sorted(vars(args).keys()):
        v = vars(args)[k]
        print('\t {} \t : \t {}'.format(k,v))
    print('=================')

    paths = {'path_image_prev': args.frame0,
             'path_image_current': args.frame1,
             'path_image_next': args.frame2,
             'path_flow_fwd': args.flow_fwd,
             'path_flow_bwd': args.flow_bwd,
             'path_backflow_bwd': args.backflow_bwd,
             'path_backflow_fwd': args.backflow_fwd,
             'path_flow_fwd_gt': args.flow_fwd_gt,
             'path_rigidity': args.rigidity,
             'path_rigidity_gt': args.rigidity_gt}

    # If tempdir does not exist yet, create it.
    if not os.path.isdir(args.tempdir):
        os.makedirs(args.tempdir)

    # Call the computation with the provided paths
    compute_mrflow(paths,args)


if __name__ == '__main__':
    main()







