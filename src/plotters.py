import wandb
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import torch
from matplotlib import collections  as mc


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
    
def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    
def plot_2D_lines_p0(step, D, z, x_1, lims=((-4, 2), (-1, 4)), x_scatter_alpha=1):
    fig,axes = plt.subplots(1, 1, figsize=(5,5),squeeze=True,sharex=True,sharey=True)

    x_0 = D.sample_marginal(250)
    axes.scatter(z[:, 0], z[:, 1],  s=60,c='white', edgecolors='black', zorder=3, label = r'$x\sim p$', alpha=1)
    axes.scatter(x_1[:, 0], x_1[:, 1], s=60,c='grey', edgecolors='black', zorder=3, label = r'$y\sim q$', alpha=1)
    axes.scatter(x_0[:, 0], x_0[:, 1], s=60, c='#ae00deff', edgecolors='black', zorder=3, label = r'$\hat{x}\sim p_{\omega}(x)$')
    y_pred = D(x_0)
    axes.scatter(y_pred[:, 0], y_pred[:, 1], s=60,c="#ff866dff", edgecolors="black", label = r'$\hat{y}\sim \gamma_{\theta,\omega}(\cdot|x)$', zorder=3)
    
    lines =  list(zip(x_0[:100].detach().cpu().numpy().astype('float'), 
                      y_pred[:100].detach().cpu().numpy().astype('float')))
    lc = mc.LineCollection(lines, linewidths=0.5, color='black')
    axes.add_collection(lc)

    axes.set_xlim(*lims[0])
    axes.set_ylim(*lims[1])
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.grid(True)
    fig.tight_layout(pad=0.5)
    
    return fig