from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.cm as cm



def draw_heat_w():
    font = {'family': 'Times New Roman', 'weight': 'bold'}

    fig, axes_l = plt.subplots()
    fig.set_size_inches(7, 4)

    a = [0.88, 0.90, 0.89, 0.90, 0.90]
    b = [0.85, 0.88, 0.88, 0.88, 0.90]
    c = [0.83, 0.89, 0.89, 0.89, 0.89]
    d = [0.77, 0.72, 0.89, 0.87, 0.87]
    e = [0.72, 0.76, 0.87, 0.71, 0.72]

    dataf = np.array([a, b, c, d, e])

    # 创建一个带有透明度的颜色映射
    cmap = cm.get_cmap('PuBu').copy()
    cmap.set_under(color='white', alpha=0.0)  # 设置最小值以下颜色为全透明的白色
    cmap.set_over(color='blue', alpha=0.2)  # 设置最大值以上颜色为半透明的蓝色
    cmap.set_bad(color='red', alpha=0)      # 设置无效数据的颜色（如果有NaN值）

    axes_l.set_ylabel('M', fontsize=15)
    datalist_shortname = ["6.0", "5.0", "4.0", "3.0", "2.0"]
    axes_l.set_yticks(range(len(datalist_shortname)))
    axes_l.set_yticklabels(datalist_shortname, fontsize=15, fontdict=font)

    axes_l.set_xlabel('N', fontsize=15)
    axes_l.set_xticks([0, 1, 2, 3, 4])
    axes_l.set_xticklabels([2.0, 3.0, 4.0, 5.0, 6.0], fontsize=15, fontdict=font)

    # 设置边框
    for edge, spine in axes_l.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.3)
        spine.set_edgecolor('black')

    norm = mcolors.Normalize(vmin=0.70, vmax=0.95)  # 设置规范化的最小和最大值
    im = axes_l.imshow(dataf, cmap=cmap, norm=norm, aspect=0.5)

    for m in range(dataf.shape[0]):
        for n in range(dataf.shape[1]):
            value = dataf[m, n]
            text_color = "black" if value < 0.85 else "white"
            text_value = f'{value:.2f}' if not np.isnan(value) else '-'
            axes_l.text(n, m, text_value, ha='center', va='center', color=text_color, fontsize=15, fontdict=font)

    cax2 = fig.add_axes([0.77, 0.255, 0.03, 0.77 * axes_l.get_position().height])
    clb2 = fig.colorbar(im, ax=axes_l, fraction=0.03, pad=0.1, cax=cax2)
    for l in clb2.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_fontweight('bold')
        l.set_fontsize(14)
    clb2.update_ticks()

    fig.subplots_adjust(left=0.076, bottom=0.258, right=0.813, top=0.775, wspace=0.1, hspace=0.1)

    plt.show()
    fig.savefig("wo_cond.png", bbox_inches='tight')
    fig.savefig("m_n_hyper.pdf", bbox_inches='tight')

draw_heat_w()

