import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "DP-GAN", "PDP-Diffusion", "DP-LDM", "PrivImage"]

def plot_con_uncon_fig(ax, data1, label1, data2, label2, xlabel, yticks=True):
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
    ax.set_xticks([0.0,20.0,40.0,60.0,80.0]) 
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
    

fig = plt.figure(figsize=(9.0, 4.0))
axs = fig.subplots(1, 2)

colors= ['#A1A9D0', '#2F7FC1']

accs_fmnist_imagenet = np.array([70.1, 68.8, 72.2, 56.2, 81.7, 48.8, 85.9])
accs_fmnist_places365 = np.array([69.5, 55.9, 75.5, 45.5, 81.1, 59.3, 82.4])

flds_fmnist_imagenet = np.array([26.0, 27.4, 18.0, 36.6, 8.6, 30.2, 5.3])
flds_fmnist_places365 = np.array([26.1, 50.8, 18.6, 38.9, 8.9, 24.3, 7.8])

plot_pre_nonpre_fig(axs[0], accs_fmnist_imagenet, 'imagenet', accs_fmnist_places365, 'places365', 'Acc (%)')
plot_pre_nonpre_fig(axs[1], flds_fmnist_imagenet, 'imagenet', flds_fmnist_places365, 'places365', 'FLD', yticks=False)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_place_imagenet.png", bbox_inches='tight')
fig.savefig("fmnist_place_imagenet.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

accs_cifar10_imagenet = np.array([22.3, 20.0, 30.2, 20.6, 27.4, 21.3, 66.5])
accs_cifar10_places365 = np.array([23.6, 20.6, 29.7, 18.3, 24.6, 18.9, 55.5])

flds_cifar10_imagenet = np.array([31.5, 41.2, 30.4, 31.0, 14.7, 16.0, 7.3])
flds_cifar10_places365 = np.array([28.6, 52.9, 32.7, 35.3, 26.7, 35.1, 9.8])

plot_con_uncon_fig(axs[0], accs_cifar10_imagenet, 'imagenet', accs_cifar10_places365, 'places365', 'Acc (%)')
plot_con_uncon_fig(axs[1], flds_cifar10_imagenet, 'imagenet', flds_cifar10_places365, 'places365', 'FLD', yticks=False)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("cifar10_place_imagenet.png", bbox_inches='tight')
fig.savefig("cifar10_place_imagenet.pdf", bbox_inches='tight')
