#! /usr/bin/env python

import numpy as np
import tempfile
import subprocess
import os,sys
import shutil # for rmtree
import time

from skimage import io as sio
from skimage import img_as_ubyte
from flow_io import flow_read
import caching

cpath = os.path.split(os.path.abspath(__file__))[0]
PATH_EPIC = os.path.join(cpath, '..', 'extern', 'epic')


def _compute_epicflow(I1, I2, edgefile=None, matchfile=None, preset='sintel'):
    """ Internal function to compute EPIC flow.

    Moved into a separate function to be able to use caching.
    """

    path_sed = os.path.join(PATH_EPIC, 'sed', 'release')
    path_toolbox = os.path.join(PATH_EPIC, 'sed', 'toolbox')
    path_deepmatching = os.path.join(PATH_EPIC, 'deepmatching/deepmatching_1.2.2_c++')
    path_epic_bin = os.path.join(PATH_EPIC, 'EpicFlow_v1.00')

    tempdir = tempfile.mkdtemp()


    if matchfile is None:
        matchfile = os.path.join(tempdir, 'matches.txt')
    if edgefile is None:
        edgefile = os.path.join(tempdir, 'edges.dat')


    # Save images into temp dir
    if type(I1) == str:
        path_I1 = I1
    else:
        path_I1 = os.path.join(tempdir, 'image1.png')
        sio.imsave(path_I1, img_as_ubyte(I1))

    if type(I2) == str:
        path_I2 = I2
    else:
        path_I2 = os.path.join(tempdir, 'image2.png')
        sio.imsave(path_I2, img_as_ubyte(I2))

    # We need an output file for epic flow.
    path_flow = os.path.join(tempdir, 'flow.flo')

    #
    # Compute edges
    #
    if (not os.path.isfile(edgefile)) or os.path.getsize(edgefile) < 10:
        sys.stdout.write('\nComputing edges...')
        sys.stdout.flush()
        te0 = time.time()
        call = ['matlab',
                '-nodesktop',
                '-nojvm',
                '-r', "addpath('{0}'); addpath(genpath('{1}')); load('{2}/modelFinal.mat'); I = uint8(imread('{3}')); if size(I,3)==1, I = cat(3,I,I,I); end; edges = edgesDetect(I, model); fid=fopen('{4}','wb'); fwrite(fid,transpose(edges),'single'); fclose(fid); exit".format(path_sed, path_toolbox, path_epic_bin, path_I1, edgefile)]

        with open(os.devnull, 'w') as dnull:
            #subprocess.call(call, stdout=dnull,stderr=subprocess.STDOUT)
            subprocess.call(call)
        te1 = time.time()
        sys.stdout.write('done. Took {} sec'.format(te1-te0))
        sys.stdout.flush()
        assert(os.path.isfile(edgefile))
    else:
        print('Using edges from {}'.format(edgefile))

    #
    # Compute deep matches
    #
    if (not os.path.isfile(matchfile)) or os.path.getsize(matchfile) < 10:
        sys.stdout.write('\nComputing matches...')
        sys.stdout.flush()
        tm0 = time.time()

        call = [os.path.join(path_deepmatching, 'deepmatching'),
                path_I1, 
                path_I2]

        with open(matchfile, 'w') as mfile:
            subprocess.call(call, stdout=mfile)

        tm1 = time.time()

        sys.stdout.write('done. Took {} sec'.format(tm1-tm0))
        sys.stdout.flush()

        assert (os.path.isfile(matchfile))

    else:
        print('Using matches from {}'.format(matchfile))

    #
    # Compute EPIC Flow
    #
    sys.stdout.write('\nComputing flow...')
    sys.stdout.flush()
    if preset == 'sintel':
        pflag = '-sintel'
    elif preset == 'kitti':
        pflag = '-kitti'

    tf0 = time.time()
    call = [os.path.join(path_epic_bin, 'epicflow'),
        path_I1,
        path_I2,
        edgefile,
        matchfile,
        path_flow,
        pflag]
    subprocess.call(call)
    tf1 = time.time()
    sys.stdout.write('done. Took {} sec'.format(tf1-tf0))
    sys.stdout.flush()

    assert (os.path.isfile(path_flow))

    # Finally, read the flow back again and return.
    u,v = flow_read(path_flow)

    # Delete temp dir
    #shutil.rmtree(tempdir)

    return u,v




def compute_epicflow(I1, I2, edgefile=None, matchfile=None, preset='sintel', use_caching=True):
    """Compute EPICFlow using the official code release.

    Parameters
    ----------
    I1, I2 : ndarray
             Input images.

    edgefile : str, optional
               Path to pre-computed edge file. If not given,
               it is recomputed. If it is given but does not
               exist, the edgefile is saved in this path.
    matchfile : str, optional
                Path to pre-computed match file.
                matchfile == None, the matches are recomputed.
                If matchfile is given but does not exist, the
                matches are saved in this position.
    use_caching : bool, optional
                Used cached computation based on an md5sum of
                the input frames.

    Returns
    -------
    u,v : ndarray
          Output flow.

    """

    if use_caching:
        u,v = caching.call_npy(
                os.path.join(os.environ['MRFLOW_HOME'],'cache'), # Caching dir
                np.vstack((I1,I2)).tostring() + preset, # Caching data string
                _compute_epicflow, # Function to call
                I1, I2, # From here on: parameters
                edgefile=edgefile,
                matchfile=matchfile,
                preset=preset)
    else:
        u,v = _compute_epicflow(I1, I2, edgefile=edgefile, matchfile=matchfile, preset=preset)

    return u,v





