import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(24.0, 6.5))
axs = fig.subplots(2, 6)

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-LoRA", "PrivImage"]
eps = ['0.2', '1.0', '5.0', '10', '15', '20']
colors= ['#A1A9D0', '#D76364','#B883D4','#9E9E9E','#05B9E2','#F1D77E','#B1CE46','#8E8BFE','#FEB2B4','#FF8C42','#6DCFF6','#2F7FC1']
markers=['o', 'v', 's', 'P', 'X', 'D', '^', '*', 'H', '>', '+', 'p', '<']

accs = '''69.1	66.3	63.4	60.8	65.0	63.1
20.3	64.4	70.7	74.2	77.2	77.5
70.8	69.0	69.2	70.1	60.8	70.0
33.3	50.8	61.6	57.8	56.4	58.8
44.9	52.7	58.7	56.8	59.3	58.2
49.3	72.8	75.1	70.3	79.5	78.7
59.7	76.4	83.5	85.6	86.5	87.9
60.8	79.0	83.0	85.4	85.6	86.5
60.5	76.8	81.2	82.0	81.7	81.7
45.5	63.8	81.3	86.3	86.6	87.1
42.2	63.5	80.9	83.8	83.9	85.2
62.7	79.9	85.2	87.1	87.3	88.9'''

fids = '''138.4	66.3	100.0	106.4	104.2	98.1
311.3	253.1	200.3	120.5	106.6	103.3
61.8	63.4	72.8	74.2	73.9	72.6
95.3	29.4	23.4	23.1	24.8	25.0
179.4	99.4	100.0	93.6	94.5	100.4
150.6	74.8	41.7	77.0	28.6	30.6
122.9	28.8	23.5	17.1	14.8	13.2
31.9	16.1	7.9	6.2	5.6	5.3
45.1	32.2	23.1	20.1	18.8	18.1
181.2	118.6	78.3	53.1	53.9	50.3
204.9	95.1	49.5	43.6	40.1	37.8
29.0	12.1	6.4	5.3	4.8	4.5'''

iss = '''2.87	2.93	2.85	2.86	2.76	2.81
1.43	1.52	2.38	3.06	3.10	2.90
3.49	3.54	3.33	3.45	3.35	3.23
7.25	5.65	5.48	5.37	5.68	5.68
1.98	2.90	2.99	3.06	3.03	3.02
2.82	3.51	3.71	3.60	3.69	3.83
2.88	2.24	3.75	3.93	3.84	3.92
4.75	4.38	4.23	4.23	4.23	4.23
5.23	4.67	4.38	4.33	4.30	4.29
2.83	3.18	3.52	3.74	3.78	3.73
2.78	3.59	3.91	3.98	3.99	4.03
4.65	4.41	4.29	4.29	4.28	4.28'''

precisions = '''0.03	0.08	0.08	0.08	0.06	0.07
0.88	0.92	0.17	0.04	0.04	0.04
0.18	0.20	0.21	0.23	0.22	0.22
0.02	0.11	0.14	0.16	0.13	0.14
0.02	0.13	0.17	0.18	0.18	0.22
0.05	0.24	0.30	0.18	0.45	0.34
0.05	0.27	0.45	0.54	0.58	0.59
0.25	0.37	0.49	0.53	0.54	0.56
0.17	0.23	0.30	0.32	0.33	0.35
0.03	0.13	0.23	0.28	0.28	0.30
0.01	0.14	0.24	0.26	0.28	0.30
0.27	0.41	0.52	0.56	0.57	0.59'''

recalls = '''0.00	0.00	0.00	0.00	0.00	0.00
0.00	0.00	0.00	0.00	0.00	0.00
0.02	0.01	0.00	0.00	0.00	0.00
0.40	0.49	0.54	0.53	0.55	0.54
0.00	0.00	0.00	0.00	0.00	0.00
0.00	0.02	0.14	0.02	0.20	0.22
0.01	0.12	0.32	0.38	0.40	0.43
0.72	0.73	0.71	0.71	0.71	0.72
0.70	0.72	0.71	0.71	0.70	0.70
0.00	0.04	0.30	0.40	0.40	0.43
0.05	0.25	0.51	0.53	0.54	0.57
0.74	0.71	0.71	0.72	0.73	0.72'''

flds = '''41.3	27.6	28.5	29.4	30.2	28.7
74.3	67.2	48.7	36.4	32.0	30.9
20.5	20.1	22.5	21.3	23.1	22.7
34.1	19.3	16.8	16.2	17.1	17.6
44.4	28.5	28.3	24.9	26.4	27.0
40.3	24.4	15.6	24.0	10.6	12.4
39.0	20.4	9.3	6.6	5.3	5.0
39.1	10.3	5.7	4.9	4.2	4.1
24.4	18.7	13.4	11.7	10.9	10.4
46.6	35.2	22.2	15.3	15.9	14.6
46.8	31.2	16.7	15.2	13.8	12.9
18.2	8.1	4.9	4.3	3.7	3.6'''

accs = [[float(acc_i) for acc_i in acc.split('\t')] for acc in accs.split('\n')]
fids = [[float(acc_i) for acc_i in acc.split('\t')] for acc in fids.split('\n')]
iss = [[float(acc_i) for acc_i in acc.split('\t')] for acc in iss.split('\n')]
precisions = [[float(acc_i) for acc_i in acc.split('\t')] for acc in precisions.split('\n')]
recalls = [[float(acc_i) for acc_i in acc.split('\t')] for acc in recalls.split('\n')]
flds = [[float(acc_i) for acc_i in acc.split('\t')] for acc in flds.split('\n')]

lw = 1.3
fontsize = 14

method_idx = [0, 1, 2, 4, 5]
for idx in method_idx:
    method = methods[idx]
    acc = accs[idx]
    axs[0, 0].plot(acc, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 0].set_xticks([i for i in range(6)], eps)
axs[0, 0].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 0].set_ylabel("Acc (%)", fontsize=fontsize)
axs[0, 0].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 0].legend(fontsize=8.5)

for idx in method_idx:
    method = methods[idx]
    fid = fids[idx]
    axs[0, 1].plot(fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 1].set_xticks([i for i in range(6)], eps)
axs[0, 1].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 1].set_ylabel("FID", fontsize=fontsize)
axs[0, 1].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 1].legend(fontsize=8.5)

for idx in method_idx:
    method = methods[idx]
    is_ = iss[idx]
    axs[0, 2].plot(is_, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 2].set_xticks([i for i in range(6)], eps)
axs[0, 2].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 2].set_ylabel("Inception Score", fontsize=fontsize)
axs[0, 2].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 2].set_yticks([1, 2, 3, 4])
axs[0, 2].legend(fontsize=8.5)

for idx in method_idx:
    method = methods[idx]
    precision = precisions[idx]
    axs[0, 3].plot(precision, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 3].set_xticks([i for i in range(6)], eps)
axs[0, 3].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 3].set_ylabel("Precision", fontsize=fontsize)
axs[0, 3].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 3].legend(fontsize=8.5)

for idx in method_idx:
    method = methods[idx]
    recall = recalls[idx]
    axs[0, 4].plot(recall, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 4].set_xticks([i for i in range(6)], eps)
axs[0, 4].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 4].set_ylabel("Recall", fontsize=fontsize)
axs[0, 4].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 4].set_yticks([0, 0.1, 0.2, 0.3])
axs[0, 4].legend(fontsize=8.5)

for idx in method_idx:
    method = methods[idx]
    fld = flds[idx]
    axs[0, 5].plot(fld, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[0, 5].set_xticks([i for i in range(6)], eps)
axs[0, 5].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[0, 5].set_ylabel("FLD", fontsize=fontsize)
axs[0, 5].tick_params(axis='both', which='major', labelsize=11.5)
axs[0, 5].legend(fontsize=8.5)


method_idx = [3, 6, 7, 8, 9 ,10, 11]
for idx in method_idx:
    method = methods[idx]
    acc = accs[idx]
    axs[1, 0].plot(acc, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 0].set_xticks([i for i in range(6)], eps)
axs[1, 0].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 0].set_ylabel("Acc (%)", fontsize=fontsize)
axs[1, 0].tick_params(axis='both', which='major', labelsize=11.5)
axs[1, 0].legend(fontsize=8.5)


for idx in method_idx:
    method = methods[idx]
    fid = fids[idx]
    axs[1, 1].plot(fid, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 1].set_xticks([i for i in range(6)], eps)
axs[1, 1].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 1].set_ylabel("FID", fontsize=fontsize)
axs[1, 1].tick_params(axis='both', which='major', labelsize=11.5)
axs[1, 1].legend(fontsize=8.5)


for idx in method_idx:
    method = methods[idx]
    is_ = iss[idx]
    axs[1, 2].plot(is_, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 2].set_xticks([i for i in range(6)], eps)
axs[1, 2].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 2].set_ylabel("Inception Score", fontsize=fontsize)
axs[1, 2].tick_params(axis='both', which='major', labelsize=11.5)


for idx in method_idx:
    method = methods[idx]
    precision = precisions[idx]
    axs[1, 3].plot(precision, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 3].set_xticks([i for i in range(6)], eps)
axs[1, 3].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 3].set_ylabel("Precision", fontsize=fontsize)
axs[1, 3].tick_params(axis='both', which='major', labelsize=11.5)



for idx in method_idx:
    method = methods[idx]
    recall = recalls[idx]
    axs[1, 4].plot(recall, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 4].set_xticks([i for i in range(6)], eps)
axs[1, 4].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 4].set_ylabel("Recall", fontsize=fontsize)
axs[1, 4].tick_params(axis='both', which='major', labelsize=11.5)
axs[1, 4].legend(loc='lower right', fontsize=8.5)



for idx in method_idx:
    method = methods[idx]
    fld = flds[idx]
    axs[1, 5].plot(fld, label=method, lw=lw, markersize=5.5, color=colors[idx], marker=markers[idx])
    axs[1, 5].set_xticks([i for i in range(6)], eps)
axs[1, 5].set_xlabel("Privacy Budget $\epsilon$", fontsize=fontsize)
axs[1, 5].set_ylabel("FLD", fontsize=fontsize)
axs[1, 5].tick_params(axis='both', which='major', labelsize=11.5)
axs[1, 5].legend(fontsize=8.5)


axs[0, 0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 2].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 3].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 4].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[0, 5].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 2].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 3].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 4].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1, 5].grid(color='lightgrey', linewidth=1.0, zorder=0)

fig.subplots_adjust(wspace=0.3, hspace=0.27)
fig.savefig("eps_change.png", bbox_inches='tight')
fig.savefig("eps_change_0117.pdf", bbox_inches='tight')
