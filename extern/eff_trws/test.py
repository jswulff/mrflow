#! /usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt

from eff_trws import Eff_TRWS

"""
A quick, trivial test example, adapted from Andreas Mueller's pygco package.
"""

lambd = 1000

def example_multinomial():
    # generate dataset with five stripes
    np.random.seed(25)
    h = 256
    w = 320
    nlabels = 5
    #h = 100
    #w = 110
    #nlabels = 5

    I = np.zeros((h, w, nlabels))
    for i in range(nlabels):
        I[:, i*(w/nlabels):(i+1)*(w/nlabels), i] = -1

    unaries = I + 8.0 * np.random.normal(size=I.shape)
    x = np.argmin(I, axis=2)
    unaries = (unaries * 100).astype(np.int32)
    unaries -= unaries.min()
    unaries = unaries + 1

    #unaries_swapped = unaries.swapaxes(0,2).swapaxes(1,2).copy()
    #print('unaries_swapped.shape : {}'.format(unaries_swapped.shape))
    #print('unaries_swapped.size : {}'.format(unaries_swapped.size))
    #np.savetxt('datacost2.txt', unaries_swapped.flatten().astype('int'),newline=' ',fmt='%d')

    x_thresh = np.argmin(unaries, axis=2)

    # potts potential
    TRWS = Eff_TRWS(w,h,nlabels)

    unaries_ = unaries.astype('int32').copy('c')
    labels_out = np.zeros((h,w),dtype='int32')
    TRWS.solve(unaries_, lambd, labels_out, use_trws=False, effective_part_opt=True)

    result = labels_out.copy()

    plt.figure()
    plt.imshow(labels_out, interpolation='nearest')
    plt.title('Normal optimization')

def example_multinomial_truncated():
    # generate dataset with five stripes
    np.random.seed(25)
    h = 256
    w = 320
    nlabels = 5
    #h = 100
    #w = 110
    #nlabels = 5

    I = np.zeros((h, w, nlabels))
    for i in range(nlabels):
        I[:, i*(w/nlabels):(i+1)*(w/nlabels), i] = -1

    unaries = I + 8.0 * np.random.normal(size=I.shape)
    x = np.argmin(I, axis=2)
    unaries = (unaries * 100).astype(np.int32)
    unaries -= unaries.min()
    unaries = unaries + 1

    #unaries_swapped = unaries.swapaxes(0,2).swapaxes(1,2).copy()
    #print('unaries_swapped.shape : {}'.format(unaries_swapped.shape))
    #print('unaries_swapped.size : {}'.format(unaries_swapped.size))
    #np.savetxt('datacost2.txt', unaries_swapped.flatten().astype('int'),newline=' ',fmt='%d')

    x_thresh = np.argmin(unaries, axis=2)

    trunc=2
    # potts potential
    TRWS = Eff_TRWS(w,h,nlabels,trunc)

    unaries_ = unaries.astype('int32').copy('c')
    labels_out = np.zeros((h,w),dtype='int32')
    TRWS.solve(unaries_, lambd, labels_out, use_trws=False, effective_part_opt=True)

    result = labels_out.copy()

    plt.figure()
    plt.imshow(labels_out, interpolation='nearest')
    plt.title('Truncated optimization')



def example_multinomial_weighted():
    # generate dataset with five stripes
    np.random.seed(25)
    h = 256
    w = 320
    nlabels = 5
    #h = 100
    #w = 110
    #nlabels = 5

    I = np.zeros((h, w, nlabels))
    for i in range(nlabels):
        I[:, i*(w/nlabels):(i+1)*(w/nlabels), i] = -1

    # Generate weights
    I_gt = np.argmin(I,axis=2)
    wy = np.vstack((np.diff(I_gt,axis=0),np.zeros((1,w))))
    wx = np.hstack((np.diff(I_gt,axis=1),np.zeros((h,1))))
    wy_weighted = np.exp(- wy**2 / ( 2 * max(1e-6,(wy**2).mean()) ) ).astype('float32')
    wx_weighted = np.exp(- wx**2 / ( 2 * max(1e-6,(wx**2).mean()) ) ).astype('float32')

    unaries = I + 8.0 * np.random.normal(size=I.shape)
    x = np.argmin(I, axis=2)
    unaries = (unaries * 100).astype(np.int32)
    unaries -= unaries.min()
    unaries = unaries + 1

    #unaries_swapped = unaries.swapaxes(0,2).swapaxes(1,2).copy()
    #print('unaries_swapped.shape : {}'.format(unaries_swapped.shape))
    #print('unaries_swapped.size : {}'.format(unaries_swapped.size))
    #np.savetxt('datacost2.txt', unaries_swapped.flatten().astype('int'),newline=' ',fmt='%d')

    x_thresh = np.argmin(unaries, axis=2)

    # potts potential
    TRWS = Eff_TRWS(w,h,nlabels)

    unaries_ = unaries.astype('int32').copy('c')
    labels_out = np.zeros((h,w),dtype='int32')
    TRWS.solve(unaries_, lambd, labels_out, weights_horizontal=wx_weighted, weights_vertical=wy_weighted, use_trws=True, effective_part_opt=True)

    result = labels_out.copy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(wx_weighted)
    plt.title('wx')
    plt.subplot(122)
    plt.imshow(wy_weighted)
    plt.title('wy')

    plt.figure()
    plt.imshow(labels_out, interpolation='nearest')
    plt.title('Weighted')

    plt.figure()
    plt.imshow(I_gt, interpolation='nearest')
    plt.title('GT')




def example_tsukuba():
    #I = plt.imread('eff_mlabel_ver1/images/tsukuba_l.ppm')
    #h,w = I.shape[:2]

    #D = np.loadtxt('eff_mlabel_ver1/images/datacost.txt').reshape((-1,h,w))
    h = 288
    w = 384
    D = np.loadtxt('./datacost2.txt')
    print(D.size)
    D = D.reshape((-1,h,w))


    # unaries
    D_restacked = np.dstack([D[i,:,:] for i in range(D.shape[0])]).astype('int32')
    #unaries = D_restacked
    unaries = D_restacked.copy('c')
    
    print(unaries)
    print(unaries.dtype)
    print(unaries.shape)
    print(unaries.min(), unaries.max())

    n_labels = unaries.shape[2]

    TRWS = Eff_TRWS(w,h,n_labels)

    #print(unaries)

    #print(unaries)
    #print(unaries.dtype)
    #print(unaries.shape)
    #print(unaries.min(), unaries.max())



    labels_out = np.zeros((h,w)).astype('int32')
    unaries_ = unaries.astype('int32').copy('c')
    print('--unaries_--')
    print(unaries_.flags)
    print(unaries_.shape)
    print(unaries_.dtype)
    print('--labels_out--')
    print(labels_out.flags)
    print(labels_out.shape)
    print(labels_out.dtype)
    print('--unaries --')
    print(unaries_.ravel()[(1*w+1)*n_labels+1])
    print(unaries_.ravel()[(15*w+10)*n_labels+5])
    TRWS.solve(unaries_, lambd, labels_out, use_trws=False, effective_part_opt=True)

    plt.figure()
    plt.imshow(labels_out)

#example_tsukuba()
example_multinomial()
example_multinomial_truncated()
#example_multinomial_weighted()
plt.show()
