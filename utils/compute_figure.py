#! /usr/bin/env python2

import numpy as np
try:
    from matplotlib import pyplot as plt
except:
    plt = None

try:
    from skimage import color,io,img_as_float,img_as_ubyte
except:
    color = None
    io = None
    img_as_float = None
    img_as_ubyte = None

import flow_viz
import flow_homography

import os,sys


def structure2image(structure, rigidity_refined,cmap='hot', structure_min=None, structure_max=None):
    if structure_min is None:
        structure_min = np.percentile(structure[rigidity_refined==1].ravel(), 2)
    if structure_max is None:
        structure_max = np.percentile(structure[rigidity_refined==1].ravel(), 98)
    Istructure = (structure - structure_min) / (structure_max-structure_min)
    Istructure = np.clip(Istructure,0,1)

    cm = plt.get_cmap(cmap)
    Istructure = cm(Istructure)[:,:,:3]*255.0
    Istructure[:,:,0][rigidity_refined==0] = 128
    Istructure[:,:,1][rigidity_refined==0] = 0
    Istructure[:,:,2][rigidity_refined==0] = 128
    return Istructure.astype('uint8')
   


# Teaser
def plot_figure_1(images, rigidity_refined, structure_refined, flow_estimated, flow_gt):
    """ Plot teaser image:
    - Triplet of frames
    - Segmentation
    - Structure
    - Flow
    """
    if not os.path.isdir('./teaser'):
        os.makedirs('teaser')

    I1 = img_as_ubyte(images[1])

    cm_bwr = plt.get_cmap('bwr')
    Irigidity = cm_bwr(rigidity_refined.astype('float32'))

    Istructure = structure2image(structure_refined, rigidity_refined)
    #Istructure_gray = structure2image(structure_refined, rigidity_refined)
    #Istructure_plasma = structure2image(structure_refined, rigidity_refined,cmap='plasma')
    #Istructure_inferno = structure2image(structure_refined, rigidity_refined,cmap='inferno')
    #Istructure_hot = structure2image(structure_refined, rigidity_refined,cmap='hot')
    #Istructure_magma =structure2image(structure_refined, rigidity_refined,cmap='magma') 
    #Istructure_viridis =structure2image(structure_refined, rigidity_refined,cmap='viridis') 
    #Istructure_jet =structure2image(structure_refined, rigidity_refined,cmap='jet') 
    #Istructure_rainbow =structure2image(structure_refined, rigidity_refined,cmap='rainbow') 

    Iflow_estimated = flow_viz.computeFlowImage(flow_estimated[0], flow_estimated[1])
    Iflow_gt = flow_viz.computeFlowImage(flow_gt[0],flow_gt[1])

    io.imsave('./teaser/01_images.png', I1)
    io.imsave('./teaser/02_rigidity.png', Irigidity)
    io.imsave('./teaser/03_structure.png', Istructure)
    #io.imsave('./teaser/03_structure_gray.png', Istructure_gray)
    #io.imsave('./teaser/03_structure_plasma.png', Istructure_plasma)
    #io.imsave('./teaser/03_structure_inferno.png', Istructure_inferno)
    #io.imsave('./teaser/03_structure_hot.png', Istructure_hot)
    #io.imsave('./teaser/03_structure_magma.png', Istructure_magma)
    #io.imsave('./teaser/03_structure_viridis.png', Istructure_viridis)
    #io.imsave('./teaser/03_structure_jet.png', Istructure_jet)
    #io.imsave('./teaser/03_structure_rainbow.png', Istructure_rainbow)
    io.imsave('./teaser/04_flowest.png', Iflow_estimated)
    io.imsave('./teaser/05_flowgt.png', Iflow_gt)

def plot_figure_2(images,
        flow_init,
        rigidity_init,
        structure_init,
        occlusions,
        rigidity_refined,
        structure_refined,
        flow_estimated):
    if not os.path.isdir('./diagram'):
        os.makedirs('diagram')

    io.imsave('./diagram/inputframe0.png', img_as_ubyte(images[0]))
    io.imsave('./diagram/inputframe1.png', img_as_ubyte(images[1]))
    io.imsave('./diagram/inputframe2.png', img_as_ubyte(images[2]))
    
    Iuvinit_bwd = flow_viz.computeFlowImage(flow_init[0][0],flow_init[0][1])
    Iuvinit_fwd = flow_viz.computeFlowImage(flow_init[1][0],flow_init[1][1])
    io.imsave('./diagram/inputflow0.png', Iuvinit_bwd)
    io.imsave('./diagram/inputflow1.png', Iuvinit_fwd)

    cm_bwr = plt.get_cmap('bwr')
    Irigidity_init = cm_bwr(rigidity_init.astype('float32'))
    Irigidity_refined = cm_bwr(rigidity_refined.astype('float32'))
    io.imsave('./diagram/rigidity_init.png', Irigidity_init)
    io.imsave('./diagram/rigidity_refined.png', Irigidity_refined)

    Istructure_init = structure2image(structure_init, rigidity_refined)
    Istructure_refined = structure2image(structure_refined, rigidity_refined)
    io.imsave('./diagram/structure_init.png', Istructure_init)
    io.imsave('./diagram/structure_refined.png', Istructure_refined)

    occ_bwd, occ_fwd = occlusions
    Iocclusions = np.ones_like(Istructure_init) * 255
    Iocclusions[:,:,0][occ_bwd>0] = 255
    Iocclusions[:,:,1][occ_bwd>0] = 0
    Iocclusions[:,:,2][occ_bwd>0] = 0
    Iocclusions[:,:,0][occ_fwd>0] = 0
    Iocclusions[:,:,1][occ_fwd>0] = 0
    Iocclusions[:,:,2][occ_fwd>0] = 255
    io.imsave('./diagram/occlusions.png', Iocclusions)

    Iuvest = flow_viz.computeFlowImage(flow_estimated[0],flow_estimated[1])
    io.imsave('./diagram/outputflow.png', Iuvest)








def plot_figure_3(image, rigidity_cnn, rigidity_motion, rigidity_structure, rigidity_refined):
    if not os.path.isdir('./rigidityestimation'):
        os.makedirs('./rigidityestimation')

    cm_bwr = plt.get_cmap('bwr')
    Irigidity_cnn = cm_bwr(rigidity_cnn.astype('float32'))
    Irigidity_motion = cm_bwr(rigidity_motion.astype('float32'))
    Irigidity_structure = cm_bwr(rigidity_structure.astype('float32'))
    Irigidity_refined = cm_bwr(rigidity_refined.astype('float32'))

    io.imsave('./rigidityestimation/01_image.png', img_as_ubyte(image))
    io.imsave('./rigidityestimation/02_rigidity_cnn.png', Irigidity_cnn)
    io.imsave('./rigidityestimation/03_rigidity_motion.png', Irigidity_motion)
    io.imsave('./rigidityestimation/04_rigidity_structure.png', Irigidity_structure)
    io.imsave('./rigidityestimation/05_rigidity_refined.png', Irigidity_refined)



def plot_figure_5(images, rigidity_refined, structure_refined, flow_estimated, flow_init, flow_gt, flow_gt_valid):
    if not os.path.isdir('./results'):
        os.makedirs('results')

    I = img_as_ubyte((images[0]+images[1]+images[2])/3.0)
    io.imsave('./results/01_image.png',I)

    cm_bwr = plt.get_cmap('bwr')
    Irigidity = cm_bwr(rigidity_refined.astype('float32'))
    io.imsave('./results/02_rigidity.png',Irigidity)

    Istructure = structure2image(structure_refined, rigidity_refined)
    io.imsave('./results/03_structure.png',Istructure)

    Iuv_est = flow_viz.computeFlowImage(flow_estimated[0],flow_estimated[1])
    io.imsave('./results/04_flow.png',Iuv_est)

    epe_est = np.sqrt((flow_estimated[0]-flow_gt[0])**2 + (flow_estimated[1]-flow_gt[1])**2)
    epe_init = np.sqrt((flow_init[0]-flow_gt[0])**2 + (flow_init[1]-flow_gt[1])**2)

    #import ipdb; ipdb.set_trace()

    epe_est[flow_gt_valid==0] = 0
    epe_init[flow_gt_valid==0] = 0

    epe_diff = epe_init - epe_est
    epe_green = np.clip(epe_diff, 0, 3)/3.0
    epe_red = np.clip(-epe_diff, 0, 3)/3.0

    Icomparison = np.zeros((rigidity_refined.shape[0],rigidity_refined.shape[1],3))

    print(Icomparison.shape)
    print(epe_green.shape)
    print(epe_red.shape)

    Icomparison[:,:,0] = epe_red
    Icomparison[:,:,1] = epe_green
    Icomparison = img_as_ubyte(Icomparison)
    io.imsave('./results/05_comparison.png',Icomparison)



#
# Supmat figures
#

def plot_figure_6(images, rigidity_refined, structure_refined, flow_estimated, flow_init, flow_gt, flow_gt_valid):
    if not os.path.isdir('./results_supmat/temp'):
        os.makedirs('results_supmat/temp')

    I = img_as_ubyte((images[0]+images[1]+images[2])/3.0)
    io.imsave('./results_supmat/temp/01_image.png',I)

    Iuv_gt = flow_viz.computeFlowImage(flow_gt[0], flow_gt[1])
    io.imsave('./results_supmat/temp/02_gt_flow.png', Iuv_gt)

    cm_bwr = plt.get_cmap('bwr')
    Irigidity = cm_bwr(rigidity_refined.astype('float32'))
    io.imsave('./results_supmat/temp/03_rigidity.png',Irigidity)

    Istructure = structure2image(structure_refined, rigidity_refined)
    io.imsave('./results_supmat/temp/04_structure.png',Istructure)

    Iuv_est = flow_viz.computeFlowImage(flow_estimated[0],flow_estimated[1])
    io.imsave('./results_supmat/temp/05_flow.png',Iuv_est)

    epe_est = np.sqrt((flow_estimated[0]-flow_gt[0])**2 + (flow_estimated[1]-flow_gt[1])**2)
    epe_init = np.sqrt((flow_init[0]-flow_gt[0])**2 + (flow_init[1]-flow_gt[1])**2)

    #import ipdb; ipdb.set_trace()

    epe_est[flow_gt_valid==0] = 0
    epe_init[flow_gt_valid==0] = 0

    epe_diff = epe_init - epe_est
    epe_green = np.clip(epe_diff, 0, 3)/3.0
    epe_red = np.clip(-epe_diff, 0, 3)/3.0

    Icomparison = np.zeros((rigidity_refined.shape[0],rigidity_refined.shape[1],3))

    Icomparison[:,:,0] = epe_red
    Icomparison[:,:,1] = epe_green
    Icomparison = img_as_ubyte(Icomparison)
    io.imsave('./results_supmat/temp/06_comparison.png',Icomparison)





def plot_figure_factorization_a(homographies, flow):
    # Figure 91
    PTH='./figure_factorization/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)

    y,x = np.mgrid[:flow[0][0].shape[0],:flow[0][0].shape[1]]

    # parallax
    u_bwd_res, v_bwd_res = flow_homography.get_residual_flow(x,y,flow[0][0],flow[0][1],homographies[0])
    u_fwd_res, v_fwd_res = flow_homography.get_residual_flow(x,y,flow[1][0],flow[1][1],homographies[1])
    I_parallax_bwd = flow_viz.computeFlowImage(u_bwd_res,v_bwd_res)
    I_parallax_fwd = flow_viz.computeFlowImage(u_fwd_res,v_fwd_res)
    io.imsave(PTH+'parallax_bwd.png',I_parallax_bwd)
    io.imsave(PTH+'parallax_fwd.png',I_parallax_fwd)



def plot_figure_factorization_b(images, structures, structure_optimized, rigidity_refined):
    # Figure 91
    PTH='./figure_factorization/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)

    io.imsave(PTH+'image_00.png',images[0])
    io.imsave(PTH+'image_01.png',images[1])
    io.imsave(PTH+'image_02.png',images[2])

    # Structure maps

    structure_min = np.percentile(structure_optimized[rigidity_refined==1].ravel(), 2)
    structure_max = np.percentile(structure_optimized[rigidity_refined==1].ravel(), 98)
    
    Is_bwd = structure2image(structures[0], rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)
    Is_fwd = structure2image(structures[1], rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)
    Is_comb = structure2image(structure_optimized, rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)

    io.imsave(PTH+'structure_bwd.png', Is_bwd)
    io.imsave(PTH+'structure_fwd.png', Is_fwd)
    io.imsave(PTH+'structure_comb.png', Is_comb)


def plot_figure_video_structure(structures, structure_combined, structure_optimized, rigidity_refined):
    # Figure 92
    PTH='./figure_structure/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)


    structure_min = np.percentile(structure_optimized[rigidity_refined==1].ravel(), 2)
    structure_max = np.percentile(structure_optimized[rigidity_refined==1].ravel(), 98)

    Is_fwd = structure2image(structures[1], rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)
    Is_comb = structure2image(structure_combined, rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)
    Is_opt = structure2image(structure_optimized, rigidity_refined,
                             structure_min=structure_min,
                             structure_max=structure_max)

    io.imsave(PTH+'structure_fwd.png', Is_fwd)
    io.imsave(PTH+'structure_comb.png', Is_comb)
    io.imsave(PTH+'structure_opt.png', Is_opt)

def plot_figure_video_rigidity_example(image, rigidity):
    # Figure 93
    PTH='./figure_rigidity_example/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)

    I_bw = color.rgb2gray(image)
    I_bw = np.dstack((I_bw,I_bw,I_bw))*0.5

    I_bw[:,:,0][rigidity==1] += 0.5
    I_bw[:,:,2][rigidity==0] += 0.5

    io.imsave(PTH+'image.png', image)
    io.imsave(PTH+'rigidity.png', I_bw)

def plot_figure_video_pasted_example(rigidity, flow_discrete, flow_ours):
    # Figure 94
    PTH='./figure_pasted/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)

    I_rigidity = np.dstack((rigidity,rigidity,rigidity)).astype('float')
    I_df = flow_viz.computeFlowImage(flow_discrete[0],flow_discrete[1])
    I_struc = flow_viz.computeFlowImage(flow_ours[0],flow_ours[1])

    I_struc_filtered = I_rigidity*I_struc

    I_final = I_struc_filtered + (1-I_rigidity)*I_df

    I_rigidity_ = I_rigidity.copy()
    I_rigidity_[:,:,1] = 0
    I_rigidity_[:,:,2] = 1-I_rigidity_[:,:,2]

    io.imsave(PTH+'rigidiyt.png', I_rigidity_)
    io.imsave(PTH+'discreteflow.png', I_df)
    io.imsave(PTH+'structureflow.png', I_struc)
    io.imsave(PTH+'structureflow_filtered.png', I_struc_filtered.astype('uint8'))
    io.imsave(PTH+'mrflow.png', I_final.astype('uint8'))


def plot_figure_95(images, rigidity, structure, flow_init, flow):
    # Results figure for video.

    PTH='./figure_results/'
    if not os.path.isdir(PTH):
        os.makedirs(PTH)

    # Save frame triplet
    io.imsave(PTH+'image_0.png', images[0])
    io.imsave(PTH+'image_1.png', images[1])
    io.imsave(PTH+'image_2.png', images[2])

    I_rigidity = np.dstack((rigidity,rigidity,rigidity)).astype('float')
    I_rigidity[:,:,1] = 0
    I_rigidity[:,:,2] = 1-I_rigidity[:,:,2]
    io.imsave(PTH+'rigidity.png', I_rigidity)

    I_structure = structure2image(structure, rigidity)
    io.imsave(PTH+'structure.png', I_structure)

    I_mrflow = flow_viz.computeFlowImage(flow[0],flow[1])
    I_discreteflow = flow_viz.computeFlowImage(flow_init[0],flow_init[1])
    io.imsave(PTH+'mrflow.png', I_mrflow)
    io.imsave(PTH+'discreteflow.png', I_discreteflow)



