import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "GS-WGAN", "DP-GAN", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-LoRA", "PrivImage"]

def plot_con_uncon_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
    diff = data1 - data2
    diff = diff[::-1]
    y = np.array([i for i in range(len(data1))])

    ax.barh(y[diff<0], data1[::-1][diff<0], label=label1, color='#FC8002', zorder=2)
    ax.barh(y[diff>=0], data1[::-1][diff>=0], color='#FC8002', zorder=1)

    ax.barh(y[diff<0], data2[::-1][diff<0], label=label2, color='#FABB6E', zorder=1)
    ax.barh(y[diff>=0], data2[::-1][diff>=0], color='#FABB6E', zorder=2)
    if yticks:
        ax.set_yticks(range(len(data2)), methods[::-1], fontsize=14)
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    x_max = np.where(data1>data2, data1, data2)
    diff = diff[::-1]
    for i in range(len(diff)):
        improve = diff[i]
        x = max(data1[i], data2[i])
        y = len(diff) - i - 1 - 0.12
        improve = str(round(improve, 1))
        if improve[0] != '-':
            improve = '+' + improve
        ax.text(x, y, str(improve), fontsize=14)

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
        ax.set_yticks(range(len(data2)), methods[::-1], fontsize=14)
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks([0.0,20.0,40.0,60.0,80.0, 100.0]) 
    ax.tick_params(axis='both', which='major', labelsize=14)

    x_max = np.where(data1>data2, data1, data2)
    diff = diff[::-1]
    for i in range(len(diff)):
        improve = diff[i]
        x = max(data1[i], data2[i])
        y = len(diff) - i - 1 - 0.12
        improve = str(round(improve, 1))
        if improve[0] != '-':
            improve = '+' + improve
        ax.text(x, y, str(improve), fontsize=14)
    

fig = plt.figure(figsize=(10.5, 5.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_pretrain = np.array([71.2, 66.6, 77.1, 62.1, 71.1, 85.4, 81.6, 86.3, 83.8, 87.1])
accs_nonpretrain = np.array([70.0, 67.8, 75.9, 48.4, 66.3, 85.4, 15.9, 16.3, 15.5, 85.6])

flds_pretrain = np.array([27.3, 36.2, 17.7, 28.7, 21.8, 4.9, 11.7, 15.4, 14.8, 4.3])
flds_nonpretrain = np.array([29.2, 36.4, 21.3, 28.1, 23.9, 6.6, 120.1, 118.3, 124.6, 4.9])

plot_pre_nonpre_fig(axs[0], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'Acc (%)')
plot_pre_nonpre_fig(axs[1], flds_pretrain, 'w/ pretrain', flds_nonpretrain, 'w/o pretrain', 'FLD', yticks=False)
axs[1].set_xticks([0.0,30.0,60.0,90.0,120.0,150.0])

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_wo_pretrain.png", bbox_inches='tight')
fig.savefig("fmnist_wo_pretrain.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

accs_condi = np.array([71.2, 66.6, 77.1, 62.1, 71.1, 85.4, 81.6, 86.3, 83.8, 87.1])
accs_uncondi = np.array([62.6, 56.3, 75.4, 61.4, 20.1, 84.3, 74.2, 84.6, 83.6, 79.5])

flds_condi = np.array([27.3, 36.2, 17.7, 28.7, 21.8, 4.9, 11.7, 15.4, 14.8, 4.3])
flds_uncondi = np.array([30.6, 42.2, 19.4, 34.3, 67.4, 5.5, 14.6, 17.5, 14.0, 6.1 ])

plot_con_uncon_fig(axs[0], accs_condi, 'cond.', accs_uncondi, 'uncond.', 'Acc (%)')
axs[0].set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 
plot_con_uncon_fig(axs[1], flds_condi, 'cond.', flds_uncondi, 'uncond.', 'FLD', yticks=False)
axs[1].legend(fontsize=11)
axs[1].set_xticks([0.0,20.0,40.0,60.0,80.0])

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_wo_cond.png", bbox_inches='tight')
fig.savefig("fmnist_wo_cond.pdf", bbox_inches='tight')
