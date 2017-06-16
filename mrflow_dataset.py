#! /usr/bin/env python2

import sys,os
import argparse
import subprocess

MRFLOW_HOME = os.environ['MRFLOW_HOME']
sys.path.append(MRFLOW_HOME)

import dataset_parameters


def generate_paths(dataset, arg):
    """ Generate paths for either Sintel or Kitti

    Requires the following environment variables to be set:
    - SINTEL_HOME
    - KITTI_HOME
    - MRFLOW_SINTEL_INIT
    - MRFLOW_KITTI_INIT
    """
    if dataset == 'sintel':
        testtrain, pas, seq, frame = arg.split(',')
        frame = int(frame)

        print('Calling Sintel preparation with')
        print('\t TESTTRAIN = {}'.format(testtrain))
        print('\t PASS = {}'.format(pas))
        print('\t SEQ = {}'.format(seq))
        print('\t FRAME = {}'.format(frame))

        # Build paths for Sintel
        SINTEL_HOME = os.environ['SINTEL_HOME']
        PPPFLOW_SINTEL_INIT = os.environ['MRFLOW_SINTEL_INIT']

        path_image_prev = os.path.join(SINTEL_HOME, testtrain, pas, seq, 'frame_{0:04d}.png'.format(frame-1))
        path_image_current = os.path.join(SINTEL_HOME, testtrain, pas, seq, 'frame_{0:04d}.png'.format(frame))
        path_image_next = os.path.join(SINTEL_HOME, testtrain, pas, seq, 'frame_{0:04d}.png'.format(frame+1))

        # Flow from reference frame to adjacent frames
        path_flow_fwd = os.path.join(PPPFLOW_SINTEL_INIT, 'flow', testtrain, pas, seq, 'frame_{0:04d}_fwd.flo'.format(frame))
        path_flow_bwd = os.path.join(PPPFLOW_SINTEL_INIT, 'flow', testtrain, pas, seq, 'frame_{0:04d}_bwd.flo'.format(frame))

        # Flow from adjacent frames back to reference frame
        path_backflow_fwd = os.path.join(PPPFLOW_SINTEL_INIT, 'flow', testtrain, pas, seq, 'frame_{0:04d}_bwd.flo'.format(frame+1))
        path_backflow_bwd = os.path.join(PPPFLOW_SINTEL_INIT, 'flow', testtrain, pas, seq, 'frame_{0:04d}_fwd.flo'.format(frame-1))

        # Estimated rigidity
        path_rigidity = os.path.join(PPPFLOW_SINTEL_INIT, 'rigidity', testtrain, pas, seq, 'frame_{0:04d}.png'.format(frame))

        # Add GT regions if we are in training pass
        if testtrain == 'training':
            path_flow_fwd_gt = os.path.join(SINTEL_HOME, testtrain, 'flow', seq, 'frame_{0:04d}.flo'.format(frame))
            path_rigidity_gt = os.path.join(SINTEL_HOME, testtrain, 'rigidity', seq, 'frame_{0:04d}.png'.format(frame))
        else:
            path_flow_fwd_gt = ''
            path_rigidity_gt = ''

    elif dataset == 'kitti': 
        testtrain, frame = arg.split(',')
        frame = int(frame)

        # KITTI
        print('Calling KITTI preparation with')
        print('\t TESTTRAIN = {}'.format(testtrain))
        print('\t FRAME = {}'.format(frame))


        # Build paths for Sintel
        KITTI_HOME = os.environ['KITTI_HOME']
        PPPFLOW_KITTI_INIT = os.environ['MRFLOW_KITTI_INIT']

        # Hack for file layout
        if testtrain == 'training':
            testtrain_ = 'training'
        else:
            testtrain_ = 'testing'

        path_image_prev = os.path.join(KITTI_HOME, testtrain_, 'image_2', '{0:06d}_09.png'.format(frame))
        path_image_current = os.path.join(KITTI_HOME, testtrain_, 'image_2', '{0:06d}_10.png'.format(frame))
        path_image_next = os.path.join(KITTI_HOME, testtrain_, 'image_2', '{0:06d}_11.png'.format(frame))

        # Flow from reference frame to adjacent frames
        path_flow_fwd = os.path.join(PPPFLOW_KITTI_INIT, 'flow', testtrain, '{0:06d}_10_fwd.flo'.format(frame))
        path_flow_bwd = os.path.join(PPPFLOW_KITTI_INIT, 'flow', testtrain, '{0:06d}_10_bwd.flo'.format(frame))

        # Flow from adjacent frames back to reference frame
        path_backflow_bwd = os.path.join(PPPFLOW_KITTI_INIT, 'flow', testtrain, '{0:06d}_09_fwd.flo'.format(frame))
        path_backflow_fwd = os.path.join(PPPFLOW_KITTI_INIT, 'flow', testtrain, '{0:06d}_11_bwd.flo'.format(frame))

        # Estimated rigidity
        path_rigidity = os.path.join(PPPFLOW_KITTI_INIT, 'rigidity', testtrain, '{0:06d}_10.png'.format(frame))

        # Add GT regions if we are in training pass
        if testtrain == 'training':
            path_flow_fwd_gt = os.path.join(KITTI_HOME, testtrain, 'flow_occ', '{0:06d}_10.png'.format(frame))
            path_rigidity_gt = os.path.join(KITTI_HOME, testtrain, 'rigidity_generated', '{0:06d}_10.png'.format(frame))
        else:
            path_flow_fwd_gt = ''
            path_rigidity_gt = ''


    paths = {
        '--flow_fwd': path_flow_fwd,
        '--flow_bwd': path_flow_bwd,
        '--backflow_fwd': path_backflow_fwd,
        '--backflow_bwd': path_backflow_bwd,
        '--rigidity': path_rigidity
        }

    if path_flow_fwd_gt:
        paths['--flow_fwd_gt'] = path_flow_fwd_gt
    if path_rigidity_gt:
        paths['--rigidity_gt'] = path_rigidity_gt

    paths_images = [path_image_prev, path_image_current, path_image_next]

    return paths,paths_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Dataset to use (sintel/kitti)')
    parser.add_argument('token', type=str, help='Token determining the frame.\n For KITTI, please give as {training/test},frame.\n For Sintel, give as {training/test},pass,seq,frame.')
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    paths,paths_images = generate_paths(args.dataset,args.token)
    if args.dataset == 'kitti':
        testtrain, frame = args.token.split(',')
        params_default = dataset_parameters.kitti_parameters
        params_default['tempdir'] = os.path.join('data_seqs', testtrain, '{0:06d}'.format(int(frame)))
    elif args.dataset == 'sintel':
        testtrain, pas, seq, frame = args.token.split(',')
        params_default = dataset_parameters.sintel_parameters
        params_default['tempdir'] = os.path.join('data_seqs', testtrain, pas, seq, 'frame_{0:04d}'.format(int(frame)))

    # If tempdir does not exist yet, create it.
    if not os.path.isdir(params_default['tempdir']):
        os.makedirs(params_default['tempdir'])

    # Set up params to call mr-flow with
    args_mrflow = {}
    for k,v in params_default.items():
        args_mrflow['--' + k] = str(v)

    for k,v in paths.items():
        args_mrflow[k] = v

    remainder_args = zip(args.args[::2],args.args[1::2])
    for k,v in remainder_args:
        args_mrflow[k] = v

    args_mrflow_array = []
    for k,v in args_mrflow.items():
        args_mrflow_array.append(k)
        args_mrflow_array.append(v)

    args_mrflow_array.append(paths_images[0])
    args_mrflow_array.append(paths_images[1])
    args_mrflow_array.append(paths_images[2])

    print('Calling MR-Flow with arguments: ')
    for k,v in zip(args_mrflow_array[::2],args_mrflow_array[1::2]):
        print('\t{}\t:\t{}'.format(k,v))
    print('')

    subprocess.call(['python', 'mrflow.py',] + args_mrflow_array)


if __name__ == '__main__':
    main()







