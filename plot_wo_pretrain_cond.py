import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "GS-WGAN", "DP-GAN", "PDP-Diffusion", "DP-LDM", "PrivImage"]

def plot_one_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
    diff = data1 - data2
    y = np.array([i for i in range(len(data1))])

    ax.barh(y[diff<0], data1[::-1][diff<0], label=label1, color='#89CFE6', zorder=1)
    ax.barh(y[diff>=0], data1[::-1][diff>=0], color='#89CFE6', zorder=2)

    ax.barh(y[diff<0], data2[::-1][diff<0], label=label2, color='#129ECC', zorder=2)
    ax.barh(y[diff>=0], data2[::-1][diff>=0], color='#129ECC', zorder=1)
    ax.legend()
    if yticks:
        ax.set_yticks(range(len(data2)), methods[::-1])
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel)

fig = plt.figure(figsize=(8.0, 4.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_pretrain = np.array([29.0, 28.2, 25.1, 21.3, 30.5, 36.8, 15.1, 36.8])
accs_nonpretrain = np.array([22.3, 20.8, 24.2, 24.2, 19.5, 27.4, 21.3, 66.7])

plot_one_fig(axs[0], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'Acc (%)')
plot_one_fig(axs[1], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'FID', yticks=False)

fig.subplots_adjust(wspace=0.18, hspace=0.3)
fig.suptitle('w/ pretrain vs w/o pretrain')
fig.savefig("wo_pretrain.png", bbox_inches='tight')
fig.savefig("wo_pretrain.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

plot_one_fig(axs[0], accs_pretrain, 'cond.', accs_nonpretrain, 'uncond.', 'Acc (%)')
plot_one_fig(axs[1], accs_pretrain, 'cond.', accs_nonpretrain, 'uncond.', 'FID', yticks=False)

fig.subplots_adjust(wspace=0.18, hspace=0.3)
fig.suptitle('cond. vs uncond.')
fig.savefig("wo_cond.png", bbox_inches='tight')
fig.savefig("wo_cond.pdf", bbox_inches='tight')

