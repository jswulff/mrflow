# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:16:46 2015

@author: jonas
"""

import numpy as np
import flow_viz as viz
import os

have_plt=False
try:
    from matplotlib import pyplot as plt
    plt._INSTALL_FIG_OBSERVER=True
    plt.rcParams['image.cmap'] = 'jet'
    plt.figure()
    plt.close()
    have_plt=True
except:
    plt=None

def imshow(*args, **kwargs):
    if plt is None:
        return
    plt.ion()
    plt.figure()
    plt.imshow(*args, **kwargs)
    plt.show()
    plt.pause(3)

def imshow_perc(*args, **kwargs):
    if plt is None:
        return
    vmin=np.percentile(args[0],5)
    vmax=np.percentile(args[0],95)
    kwargs['vmin']=vmin
    kwargs['vmax']=vmax
    plt.ion()
    plt.figure()
    plt.imshow(*args, **kwargs)
    plt.show()
    plt.pause(3)
 

def plot(*args):
    if plt is None:
        return
    plt.ion()
    plt.figure()
    plt.plot(*args)
    plt.show()
    plt.pause(3)

def quiver(*args):
    if plt is None:
        return
    plt.ion()
    plt.figure()
    plt.quiver(*args)
    plt.show()
    plt.pause(3)


def plot_quiver(pt, uv, title, masks=None, norm=-1, outpath='.'):
    if plt is None:
        return
    plt.ioff()
    if masks is None:
        masks = [np.ones(pt.shape[0])>0,]
    if norm > 0:
        uvlen = np.sqrt((uv**2).sum(axis=1))
        uv[uvlen<norm,:] /= (1.0/norm) * uvlen[uvlen<norm][:,np.newaxis]
    colors = ['r','b','g','c','y']
    plt.figure()
    for i,m in enumerate(masks):
        plt.quiver(pt[m,0],
                pt[m,1],
                uv[m,0],
                uv[m,1],
                color=colors[i%len(colors)],
                angles='xy',
                scale_units='xy',
                scale=1)

    plt.axis('equal')
    plt.title(title)
    plt.ylim([pt[:,1].max(),0])
    save_figure(title, outpath)

def plot_scatter(pt, title, I=None,masks=None, outpath='.'):
    if plt is None:
        return
    plt.ioff()
    if masks is None:
        masks = [np.ones(pt.shape[0])>0,]
    colors = ['r','b','g','c','y']
    plt.figure()
    if I is not None:
        plt.imshow(I)
    for i,m in enumerate(masks):
        plt.plot(pt[m,0],
                pt[m,1],
                '.{}'.format(colors[i%len(colors)]))

    if I is not None:
        ymax = I.shape[0]
        xmax = I.shape[1]
    else:
        ymax = pt[:,1].max()
        xmax = pt[:,0].max()
        plt.axis('equal')
    plt.title(title)
    plt.ylim([ymax,0])
    plt.xlim([0,xmax])
    save_figure(title, outpath)


def show():
    if plt is None:
        return
    plt.show()

def plot_flow(u,v,title,outpath='.'):
    if plt is None:
        return
    plt.ioff()
    Iuv = viz.computeFlowImage(u,v)
    plt.figure()
    plt.imshow(Iuv)
    plt.title(title)
    save_figure(title,outpath)

def plot_image(x,title,colorbar=False,vmin=None,vmax=None,outpath='.',cmap=None):
    if plt is None:
        return
    plt.ioff()
    plt.figure()
    plt.imshow(x,interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if colorbar:
        plt.colorbar()
    save_figure(title,outpath)


def plot_plot(y, title, legends=None,outpath='.'):
    if plt is None:
        return
    if np.array(y).ndim == 1:
        y = np.array(y).reshape((-1,1))
    no_legends = legends is None
    if legends is None or len(legends) < y.shape[1]:
        legends = [''] * y.shape[1]
    plt.ioff()
    plt.figure()
    for d in range(y.shape[1]):
        plt.plot(y[:,d],label=legends[d])
    if not no_legends:
        plt.legend()
    plt.title(title)
    save_figure(title,outpath)


def save_figure(title,outpath='.'):
    if plt is None:
        return
    outdir = os.path.join(outpath, 'images')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    fname = outdir + '/' + title.replace(' ', '') + '.png'
    plt.savefig(fname, dpi=200,bbox_inches='tight', pad_inches=0)
    plt.close()
