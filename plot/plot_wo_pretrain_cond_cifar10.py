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
        ax.text(x, y, str(improve), fontsize=12)

def plot_pre_nonpre_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
    diff = data1 - data2
    diff = diff[::-1]
    y = np.array([i for i in range(len(data1))])

    ax.barh(y[diff<0], data1[::-1][diff<0], label=label1, color='#89CFE6', zorder=2)
    ax.barh(y[diff>=0], data1[::-1][diff>=0], color='#89CFE6', zorder=1)

    ax.barh(y[diff<0], data2[::-1][diff<0], label=label2, color='#129ECC', zorder=1)
    ax.barh(y[diff>=0], data2[::-1][diff>=0], color='#129ECC', zorder=2)
    if yticks:
        ax.set_yticks(range(len(data2)), methods[::-1], fontsize=14)
    else:
        ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks([0.0,20.0,40.0,60.0,80.0]) 
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
        ax.text(x, y, str(improve), fontsize=12)
    

fig = plt.figure(figsize=(10.5, 5.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_pretrain = np.array([26.1, 20.0, 24.0, 20.1, 32.1, 70.1, 69.9, 64.8, 77.2, 78.4])
accs_nonpretrain = np.array([29.0, 19.8, 32.2, 18.5, 35.2, 36.8, 15.1, 13.9, 12.6, 36.8])

flds_pretrain = np.array([28.4, 50.0, 40.0, 33.2, 25.3, 7.2, 9.0, 14.1, 9.3, 5.1])
flds_nonpretrain = np.array([32.1, 41.7, 27.2, 31.1, 22.5, 19.4, 47.1, 46.3, 45.6, 19.4])

plot_pre_nonpre_fig(axs[0], accs_pretrain, 'w/ pretrain', accs_nonpretrain, 'w/o pretrain', 'Acc (%)')
axs[0].set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 
plot_pre_nonpre_fig(axs[1], flds_pretrain, 'w/ pretrain', flds_nonpretrain, 'w/o pretrain', 'FLD', yticks=False)
axs[1].legend(fontsize=11)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("wo_pretrain.png", bbox_inches='tight')
fig.savefig("wo_pretrain.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

accs_condi = np.array([26.1, 20.0, 24.0, 18.5, 35.2, 70.1, 69.9, 64.8, 77.2, 78.4])
accs_uncondi = np.array([22.3, 18.7, 26.5, 21.1, 11.6, 48.5, 51.3, 56.8, 65.8, 44.0])

flds_condi = np.array([28.4, 50.0, 40.0, 33.2, 25.3, 7.2, 9.0, 14.1, 9.3, 5.1])
flds_uncondi = np.array([31.0, 51.3, 37.4, 32.9, 44.2, 10.8, 12.1, 14.9, 11.1, 7.2])

plot_con_uncon_fig(axs[0], accs_condi, 'cond.', accs_uncondi, 'uncond.', 'Acc (%)')
axs[0].set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 
plot_con_uncon_fig(axs[1], flds_condi, 'cond.', flds_uncondi, 'uncond.', 'FLD', yticks=False)
axs[1].legend(fontsize=11)
axs[1].set_xticks([0.0,15.0,30.0,45.0,60.0]) 

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("wo_cond.png", bbox_inches='tight')
fig.savefig("wo_cond.pdf", bbox_inches='tight')

