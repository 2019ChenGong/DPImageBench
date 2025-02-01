import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "GS-WGAN", "DP-GAN", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-LORA", "PrivImage"]

def plot_cifar(ax, data1, label1, data2, label2, xlabel, yticks=True):
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

def plot_fmnist(ax, data1, label1, data2, label2, xlabel, yticks=True):
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

accs_fmnist_imagenet = np.array([71.2, 66.6, 77.1, 62.1, 71.1, 85.4, 81.6, 86.3, 83.8, 87.1])
accs_fmnist_places365 = np.array([69.2, 68.6, 78.6, 62.3, 69.9, 85.7, 79.9, 83.2, 80.2, 84.0])

flds_fmnist_imagenet = np.array([27.3, 36.2, 17.7, 28.7, 21.8, 4.9, 11.7, 15.4, 14.8, 4.3])
flds_fmnist_places365 = np.array([30.3, 38.6, 18.2, 27.7, 24.3, 4.4, 13.1, 21.6, 19.1, 5.2])

plot_fmnist(axs[0], accs_fmnist_imagenet, 'imagenet', accs_fmnist_places365, 'places365', 'Acc (%)')
axs[0].set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 

plot_fmnist(axs[1], flds_fmnist_imagenet, 'imagenet', flds_fmnist_places365, 'places365', 'FLD', yticks=False)
axs[1].set_xticks([0.0,10.0,20.0,30.0,40.0,50.0]) 
axs[1].legend(fontsize=11)

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("fmnist_place_imagenet.png", bbox_inches='tight')
fig.savefig("fmnist_place_imagenet.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

accs_cifar10_imagenet = np.array([26.1, 20.0, 24.0, 20.1, 32.1, 70.1, 69.9, 64.8, 77.2, 78.4])
accs_cifar10_places365 = np.array([28.1, 20.8, 26.7, 20.5, 23.3, 60.1, 61.2, 51.3, 57.5, 63.3])

flds_cifar10_imagenet = np.array([28.4, 50.0, 40.0, 33.2, 25.3, 7.2, 9.0, 14.1, 9.3, 5.1])
flds_cifar10_places365 = np.array([29.1, 49.5, 30.8, 30.8, 27.3, 8.8, 12.9, 16.4, 14.7, 10.5])

plot_cifar(axs[0], accs_cifar10_imagenet, 'imagenet', accs_cifar10_places365, 'places365', 'Acc (%)')
axs[0].set_xticks([0.0,20.0,40.0,60.0,80.0,100.0]) 
plot_cifar(axs[1], flds_cifar10_imagenet, 'imagenet', flds_cifar10_places365, 'places365', 'FLD', yticks=False)
axs[1].set_xticks([0.0,15.0,30.0,45.0,60.0]) 
axs[1].legend(fontsize=11, loc='lower right')

fig.subplots_adjust(wspace=0.07, hspace=0.3)

fig.savefig("cifar10_place_imagenet.png", bbox_inches='tight')
fig.savefig("cifar10_place_imagenet.pdf", bbox_inches='tight')
