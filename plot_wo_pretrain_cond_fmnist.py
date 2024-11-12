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

    x_max = np.where(data1>data2, data1, data2)
    for i in range(len(diff)):
        improve = diff[i] / data2[i] * 100
        x = max(data1[i], data2[i])
        y = len(diff) - i - 1 - 0.12
        improve = str(int(improve)) + '%'
        if improve[0] != '-':
            improve = '+' + improve
        ax.text(x, y, str(improve), fontsize=12)

fig = plt.figure(figsize=(8.0, 4.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_pretrain = np.array([70.1, 68.8, 72.2, 56.8, 56.2, 81.1, 48.8, 85.9])
accs_nonpretrain = np.array([70.0, 61.8, 75.9, 48.4, 66.3, 85.4, 15.9, 85.4])

fids_pretrain = np.array([26.0, 46.4, 18.0, 24.0, 1, 8.6, 1, 5.3])
fids_nonpretrain = np.array([28.6, 28.4, 18.1, 23.6, 27.7, 6.6, 82.9, 6.6])

accs_cond = np.array([70.1, 68.8, 72.2, 56.8, 56.2, 81.1, 48.8, 85.9])
accs_uncond = np.array([71.4, 70.7, 75.8, 54.5, 39.2, 82.1, 48.8, 86.0])

fids_cond = np.array([26.0, 46.4, 18.0, 24.0, 1, 1, 1, 5.3])
fids_uncond = np.array([24.6, 49.2, 18.5, 22.5, 39.5, 8.6, 29.4, 1])

plot_one_fig(axs[0], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'Acc (%)')
plot_one_fig(axs[1], fids_pretrain, 'w/ pretrain', fids_nonpretrain, 'w/o pretrain', 'FID', yticks=False)

fig.subplots_adjust(wspace=0.08, hspace=0.3)
fig.suptitle('w/ pretrain vs w/o pretrain')
fig.savefig("wo_pretrain.png", bbox_inches='tight')
fig.savefig("wo_pretrain.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

plot_one_fig(axs[0], accs_pretrain, 'cond.', accs_nonpretrain, 'uncond.', 'Acc (%)')
plot_one_fig(axs[1], accs_pretrain, 'cond.', accs_nonpretrain, 'uncond.', 'FID', yticks=False)

fig.subplots_adjust(wspace=0.08, hspace=0.3)
fig.suptitle('cond. vs uncond.')
fig.savefig("wo_cond.png", bbox_inches='tight')
fig.savefig("wo_cond.pdf", bbox_inches='tight')

