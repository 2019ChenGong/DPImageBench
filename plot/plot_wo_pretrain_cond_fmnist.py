import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "GS-WGAN", "DP-GAN", "PDP-Diffusion", "DP-LDM", "PrivImage"]

def plot_con_uncon_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
    diff = data1 - data2
    diff = diff[::-1]
    y = np.array([i for i in range(len(data1))])

    ax.barh(y[diff<0], data1[::-1][diff<0], label=label1, color='#FC8002', zorder=2)
    ax.barh(y[diff>=0], data1[::-1][diff>=0], color='#FC8002', zorder=1)

    ax.barh(y[diff<0], data2[::-1][diff<0], label=label2, color='#FABB6E', zorder=1)
    ax.barh(y[diff>=0], data2[::-1][diff>=0], color='#FABB6E', zorder=2)
    ax.legend(fontsize=11)
    if yticks:
        ax.set_yticks(range(len(data2)), methods[::-1], fontsize=12)
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 
    ax.tick_params(axis='both', which='major', labelsize=12)

    x_max = np.where(data1>data2, data1, data2)
    diff = diff[::-1]
    for i in range(len(diff)):
        improve = diff[i]
        x = max(data1[i], data2[i])
        y = len(diff) - i - 1 - 0.12
        improve = str(round(improve, 1))
        if improve[0] != '-':
            improve = '+' + improve
        ax.text(x, y, str(improve), fontsize=12)

def plot_pre_nonpre_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
    diff = data1 - data2
    diff = diff[::-1]
    y = np.array([i for i in range(len(data1))])

    ax.barh(y[diff<0], data1[::-1][diff<0], label=label1, color='#89CFE6', zorder=2)
    ax.barh(y[diff>=0], data1[::-1][diff>=0], color='#89CFE6', zorder=1)

    ax.barh(y[diff<0], data2[::-1][diff<0], label=label2, color='#129ECC', zorder=1)
    ax.barh(y[diff>=0], data2[::-1][diff>=0], color='#129ECC', zorder=2)
    ax.legend(fontsize=11)
    if yticks:
        ax.set_yticks(range(len(data2)), methods[::-1], fontsize=12)
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks([0.0,20.0,40.0,60.0,80.0, 100.0]) 
    ax.tick_params(axis='both', which='major', labelsize=12)

    x_max = np.where(data1>data2, data1, data2)
    diff = diff[::-1]
    for i in range(len(diff)):
        improve = diff[i]
        x = max(data1[i], data2[i])
        y = len(diff) - i - 1 - 0.12
        improve = str(round(improve, 1))
        if improve[0] != '-':
            improve = '+' + improve
        ax.text(x, y, str(improve), fontsize=12)
    

fig = plt.figure(figsize=(9.0, 4.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_pretrain = np.array([70.1, 68.8, 72.2, 56.8, 56.2, 81.7, 48.8, 85.9])
accs_nonpretrain = np.array([70.0, 67.8, 75.9, 48.4, 66.3, 85.4, 15.9, 85.4])

flds_pretrain = np.array([26.0, 27.4, 18.0, 23.0, 36.6, 8.6, 30.2, 5.3])
flds_nonpretrain = np.array([28.6, 28.4, 18.1, 23.6, 27.7, 6.6, 82.9, 6.6])

plot_pre_nonpre_fig(axs[0], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'Acc (%)')
plot_pre_nonpre_fig(axs[1], flds_pretrain, 'w/ pretrain', flds_nonpretrain, 'w/o pretrain', 'FLD', yticks=False)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_wo_pretrain.png", bbox_inches='tight')
fig.savefig("fmnist_wo_pretrain.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

accs_condi = np.array([70.1, 68.8, 75.9, 48.4, 56.2, 79.8, 35.7, 85.9])
accs_uncondi = np.array([71.4, 70.7, 75.8, 54.5, 39.2, 82.1, 48.8, 86.0])

flds_condi = np.array([26.0, 46.4, 18.0, 24.0, 33.6, 8.8, 29.4, 5.3])
flds_uncondi = np.array([24.6, 49.2, 18.5, 22.5, 39.5, 8.6, 20.4, 5.2])

plot_con_uncon_fig(axs[0], accs_condi, 'cond.', accs_uncondi, 'uncond.', 'Acc (%)')
plot_con_uncon_fig(axs[1], flds_condi, 'cond.', flds_uncondi, 'uncond.', 'FLD', yticks=False)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_wo_cond.png", bbox_inches='tight')
fig.savefig("fmnist_wo_cond.pdf", bbox_inches='tight')
