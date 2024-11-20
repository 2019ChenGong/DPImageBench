import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(17.0, 6.0))
axs = fig.subplots(2, 3)

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM", "PrivImage"]
eps = ['0.2', '1.0', '5.0', '10', '15', '20']
colors= ['#A1A9D0', '#D76364','#B883D4','#9E9E9E','#05B9E2','#F1D77E','#B1CE46','#8E8BFE','#FEB2B4','#2F7FC1']
markers=['o', 'v', 's', 'P', 'X', 'D', '^', '*', 'H', '>', '<']

accs = [
    [69.1, 66.3, 63.4, 60.8, 65.0, 63.1],
[10.0, 64.4, 70.7, 74.2, 77.2, 77.5],
[70.8, 69.0, 69.2, 70.1, 60.8, 70.0],
[33.3, 50.8, 61.6, 57.8, 56.4, 58.8],
[44.9, 52.7, 58.7, 56.8, 59.3, 58.2],
[49.3, 72.8, 75.1, 70.3, 79.5, 78.7],
[59.7, 76.4, 83.5, 85.4, 86.5, 87.9],
[26.5, 40.1, 76.4, 81.1, 83.5, 83.8],
[25.5, 30.1, 50.8, 48.8, 53.5, 46.9],
[45.2, 73.5, 83.6, 85.9, 87.8, 88.0],
]

fids = [
    [138.4, 66.3, 100.0, 106.4, 104.2, 98.1],
[435.8, 353.1, 200.3, 120.5, 106.6, 103.3],
[61.8, 63.4, 72.8, 74.2, 73.9, 72.6],
[95.3, 29.4, 23.4, 23.1, 24.8, 25.0],
[179.4, 99.4, 100.0, 93.6, 94.5, 100.4],
[150.6, 74.8, 41.7, 77.0, 28.6, 30.6],
[122.9, 28.8, 23.5, 17.1, 14.8, 13.2],
[82.3, 48.8, 24.0, 16.1, 14.1, 12.5],
[143.1, 114.0, 102.5, 70.2, 68.6, 70.4],
[58.8, 32.0, 15.3, 11.2, 9.3, 9.0],
]

iss = [
    [2.87, 2.93, 2.85, 2.86, 2.76, 2.81],
[1.37, 1.52, 2.38, 3.06, 3.10, 2.90],
[3.49, 3.54, 3.33, 3.45, 3.35, 3.23],
[7.25, 5.65, 5.48, 5.37, 5.68, 5.68],
[1.98, 2.90, 2.99, 3.06, 3.03, 3.02],
[2.82, 3.51, 3.71, 3.60, 3.69, 3.83],
[2.88, 2.24, 3.75, 3.93, 3.84, 3.92],
[4.73, 4.03, 4.04, 4.01, 4.14, 4.11],
[4.10, 4.37, 4.22, 4.11, 4.00, 4.01],
[3.79, 3.86, 3.97, 3.98, 4.04, 4.03],
]

precisions = [
    [0.03, 0.08, 0.08, 0.08, 0.06, 0.07],
[0.10, 0.13, 0.17, 0.04, 0.04, 0.04],
[0.18, 0.20, 0.21, 0.23, 0.22, 0.22],
[0.02, 0.11, 0.14, 0.16, 0.13, 0.14],
[0.02, 0.13, 0.17, 0.18, 0.18, 0.22],
[0.05, 0.24, 0.30, 0.18, 0.45, 0.34],
[0.05, 0.27, 0.45, 0.54, 0.58, 0.59],
[0.12, 0.25, 0.38, 0.44, 0.47, 0.49],
[0.03, 0.09, 0.09, 0.14, 0.18, 0.15],
[0.23, 0.35, 0.50, 0.57, 0.60, 0.6],
]

recalls = [
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.02, 0.01, 0.00, 0.00, 0.00, 0.00],
[0.40, 0.49, 0.54, 0.53, 0.55, 0.54],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.02, 0.14, 0.02, 0.20, 0.22],
[0.01, 0.12, 0.32, 0.38, 0.40, 0.43],
[0.48, 0.53, 0.56, 0.59, 0.61, 0.61],
[0.24, 0.29, 0.38, 0.43, 0.44, 0.40],
[0.38, 0.44, 0.50, 0.53, 0.55, 0.54],
]

flds = [
    [41.3, 27.6, 28.5, 29.4, 30.2, 28.7],
[76.3, 67.2, 48.7, 36.4, 32.0, 30.9],
[20.5, 20.1, 22.5, 21.3, 23.1, 22.7],
[34.1, 19.3, 16.8, 16.2, 17.1, 17.6],
[44.4, 28.5, 28.3, 24.9, 26.4, 27.0],
[40.3, 24.4, 15.6, 24.0, 10.6, 12.4],
[39.0, 20.4, 9.3, 6.6, 5.3, 5.0],
[29.3, 23.1, 12.0, 8.6, 7.3, 7.1],
[41.5, 31.4, 50.8, 29.5, 27.4, 28.7],
[23.7, 13.6, 7.0, 5.3, 4.2, 4.2],
]

lw = 1.3

for idx in range(len(methods)):
    method = methods[idx]
    acc = accs[idx]
    axs[0, 0].plot(acc, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 0].set_xticks([i for i in range(6)], eps)
axs[0, 0].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[0, 0].set_ylabel("Acc (%)", fontsize=12.5)
axs[0, 0].tick_params(axis='both', which='major', labelsize=11.5)


for idx in range(len(methods)):
    method = methods[idx]
    fid = fids[idx]
    axs[0, 1].plot(fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 1].set_xticks([i for i in range(6)], eps)
axs[0, 1].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[0, 1].set_ylabel("FID", fontsize=12.5)
axs[0, 1].tick_params(axis='both', which='major', labelsize=11.5)

for idx in range(len(methods)):
    method = methods[idx]
    Is = iss[idx]
    axs[1, 2].plot(Is, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 2].set_xticks([i for i in range(6)], eps)
axs[1, 2].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[1, 2].set_ylabel("IS", fontsize=12.5)
axs[1, 2].set_yticks([0.0,2.0,4.0,6.0,8.0]) 
axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[1, 2].tick_params(axis='both', which='major', labelsize=11.5)


for idx in range(len(methods)):
    method = methods[idx]
    pr = precisions[idx]
    axs[1, 0].plot(pr, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 0].set_xticks([i for i in range(6)], eps)
axs[1, 0].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[1, 0].set_ylabel("Precision", fontsize=12.5)
axs[1, 0].tick_params(axis='both', which='major', labelsize=11.5)

for idx in range(len(methods)):
    method = methods[idx]
    re = recalls[idx]
    axs[1, 1].plot(re, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 1].set_xticks([i for i in range(6)], eps)
axs[1, 1].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[1, 1].set_ylabel("Recall", fontsize=12.5)
axs[1, 1].tick_params(axis='both', which='major', labelsize=11.5)

for idx in range(len(methods)):
    method = methods[idx]
    fld = flds[idx]
    axs[0, 2].plot(fld, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 2].set_xticks([i for i in range(6)], eps)
axs[0, 2].set_xlabel("Privacy Budget $\epsilon$", fontsize=12.5)
axs[0, 2].set_ylabel("FLD", fontsize=12.5)
axs[0, 2].set_yticks([0.0,20,40,60,80])
axs[0, 2].tick_params(axis='both', which='major', labelsize=11.5)

axs[0, 0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 2].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 2].grid(color='lightgrey', linewidth=1.0, zorder=0)

fig.subplots_adjust(wspace=0.18, hspace=0.3)
fig.savefig("eps_change.png", bbox_inches='tight')
fig.savefig("eps_change.pdf", bbox_inches='tight')