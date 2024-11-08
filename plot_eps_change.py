import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure(figsize=(11.0, 6.0))
axs = fig.subplots(2, 3)

methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM", "PrivImage"]

accs = [
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
    [0.6] * 6,
]

for idx in range(len(methods)):
    method = methods[idx]
    acc = accs[idx]
    axs[0, 0].plot(acc, label=method)

fig.savefig("eps_change.png")