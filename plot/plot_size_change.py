import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(10.0, 4.0))
axs = fig.subplots(1, 2)

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-LoRA", "PrivImage"]
gan_size = [3.8, 6.6, 10, 14.3, 19.4]
diffusion_size = [3.8, 11.1, 19.6, 44.2, 78.5]
colors= ['#A1A9D0', '#D76364','#B883D4','#9E9E9E','#05B9E2','#934B43','#B1CE46','#8E8BFE','#FEB2B4', '#FF8C42','#6DCFF6','#2F7FC1']
markers=['o', 'v', 's', 'P', 'X', 'D', '^', '*', 'H', '>', '+', 'p', '<']

gan_accs = '''28.8	22.7	22.2	25.4	26.6
19.8	21.6	22.2	18.8	18.8
33.2	31.2	30.2	31.3	30.6
18.5	21.8	21.7	20.7	21.0
35.2	27.8	23.1	26.1	27.0'''
gan_fids = '''175.6	210.4	188.0	207.0	204.4
438.4	435.6	417.0	436.4	434.5
193.0	204.4	198.0	195.0	189.8
209.6	205.6	223.9	205.5	236.0
111.9	139.7	133.6	132.0	144.6'''

dif_accs = '''36.8	41.0	41.0	41.5	42.8
70.1	76.2	80.1	80.2	80.7
69.9	76.0	79.1	80.3	77.8
64.8	71.2	74.2	73.4	75.0
77.2	75.0	76.2	76.3	78.5
78.4	78.2	76.0	79.5	78.3'''

dif_fids = '''110.1	113.8	110.3	103.2	103.8
18.4	11.9	12.5	10.1	8.8
19.8	13.5	13.4	10.8	10.0
47.4	31.1	29.7	26.8	26.0
27.8	23.2	22.1	19.7	19.9
13.1	13.9	14.9	13.3	13.9'''

gan_accs = [[float(acc_i) for acc_i in acc.split('\t')] for acc in gan_accs.split('\n')]
gan_fids = [[float(acc_i) for acc_i in acc.split('\t')] for acc in gan_fids.split('\n')]
dif_accs = [[float(acc_i) for acc_i in acc.split('\t')] for acc in dif_accs.split('\n')]
dif_fids = [[float(acc_i) for acc_i in acc.split('\t')] for acc in dif_fids.split('\n')]

lw = 1.3

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "GS-WGAN", "DP-GAN"]
colors= ['#A1A9D0', '#D76364','#B883D4','#05B9E2','#934B43']
markers=['o', 'v', 's', 'X', 'D',]

for idx in range(len(gan_accs)):
    method = methods[idx]
    acc = gan_accs[idx]
    axs[0].plot(gan_size, acc, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    # axs[0].set_xticks([i for i in range(6)], eps)
axs[0].set_xlabel("The size of synthesizer parameters (M)", fontsize=12.5)
axs[0].set_ylabel("Acc (%)", fontsize=14.0)
axs[0].tick_params(axis='both', which='major', labelsize=11.5)
axs[0].set_yticks([20, 24, 28, 32], ['20', '24', '28', '32'])


for idx in range(len(gan_fids)):
    method = methods[idx]
    fid = gan_fids[idx]
    axs[1].plot(gan_size, fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
axs[1].set_xlabel("The size of synthesizer parameters (M)", fontsize=12.5)
axs[1].set_ylabel("FID", fontsize=14.0)
axs[1].tick_params(axis='both', which='major', labelsize=11.5)

axs[0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0].legend()
axs[1].legend()

fig.subplots_adjust(wspace=0.2)
fig.savefig("gan_size_change.png", bbox_inches='tight')
fig.savefig("gan_size_change.pdf", bbox_inches='tight')

fig.clf()
axs = fig.subplots(1, 2)

methods = ["DPDM", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-LoRA", "PrivImage"]
colors= ['#B1CE46','#8E8BFE','#FEB2B4','#FF8C42','#6DCFF6','#2F7FC1']
markers=['*', 'H', '>', '+', 'p', '<']

for idx in range(len(dif_accs)):
    method = methods[idx]
    acc = dif_accs[idx]
    axs[0].plot(diffusion_size, acc, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    # axs[0].set_xticks([i for i in range(6)], eps)
axs[0].set_xlabel("The size of synthesizer parameters (M)", fontsize=12.5)
axs[0].set_ylabel("Acc (%)", fontsize=14.0)
axs[0].tick_params(axis='both', which='major', labelsize=11.5)


for idx in range(len(dif_fids)):
    method = methods[idx]
    fid = dif_fids[idx]
    if method in ['DP-LoRA', 'DP-LDM']:
        axs[1].plot([3.9, 11.9, 23.6, 39.7, 77.4], fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    else:
        axs[1].plot(diffusion_size, fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    # axs[1].set_xticks([i for i in range(6)], eps)
axs[1].set_xlabel("The size of synthesizer parameters (M)", fontsize=12.5)
axs[1].set_ylabel("FID", fontsize=14.0)
axs[1].tick_params(axis='both', which='major', labelsize=11.5)


axs[0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0].legend()
axs[1].legend()

fig.subplots_adjust(wspace=0.2)
fig.savefig("diffusion_size_change.png", bbox_inches='tight')
fig.savefig("diffusion_size_change.pdf", bbox_inches='tight')