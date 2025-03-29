import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from data.stylegan3.dataset import ImageFolderDataset


fontsize = 8
matplotlib.rcParams.update({'font.size': fontsize, 'font.weight': 'normal'})


def sample_real_images(data_path):
    dataset = ImageFolderDataset(data_path, use_labels=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=2000)
    for x, y in dataloader:
        x = x.to(torch.float32)
        break
    x = x.numpy()
    y = y.numpy()
    return x, y
    

def visualize_main():
    column_per_dataset = 6
    row_per_method = 1
    img_size = 32
    datasets = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "EuroSAT", "CelebA", "Camelyon"]
    methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DPGAN", "DPDM", "DP-FETA", "PDP-Diffusion", "DP-LDM (SD)", "DP-LDM", "DP-L0RA", "PrivImage", "Real"]
    width_per_patch = img_size * column_per_dataset
    height_per_patch = img_size * row_per_method

    fig = plt.figure(figsize=(13.6, 4.8))
    axs = fig.subplots(len(methods), len(datasets))

    gen_lists = [
    [
        "/DPImageBench/exp/dp-merf/mnist_28_eps10.0trainval-2024-10-19-06-42-21/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval-2024-10-20-06-27-04/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/cifar10_32_eps10.0trainval-2024-10-20-06-27-04/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/cifar100_32_eps10.0trainval-2024-10-20-06-30-09/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/eurosat_32_eps10.0trainval-2024-10-20-06-30-09/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/celeba_male_32_eps10.0trainval-2024-10-22-03-29-00/gen/gen.npz",
        "/DPImageBench/exp/dp-merf/camelyon_32_eps10.0trainval-2024-10-20-06-32-32/gen/gen.npz"
    ],  
    [
        "/DPImageBench/exp/dp-ntk/mnist_28_eps10.0trainval-2024-10-20-09-43-52/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/fmnist_28_eps10.0trainval-2024-10-20-09-45-58/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/cifar10_32_eps10.0trainval-2024-10-24-12-30-04/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/cifar100_32_eps10.0trainval-2024-10-24-12-30-04/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/eurosat_32_eps10.0trainval-2024-10-24-01-48-00/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/celeba_male_32_eps10.0trainval-2024-10-23-07-28-01/gen/gen.npz",
        "/DPImageBench/exp/dp-ntk/camelyon_32_eps10.0trainval-2024-10-23-07-26-25/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dp-kernel/mnist_28_eps10.0trainval-2024-10-20-06-44-57/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval-2024-10-20-06-44-57/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/cifar10_32_eps10.0trainval-2024-10-20-09-38-09/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/cifar100_32_eps10.0trainval-2024-10-22-14-48-27/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/eurosat_32_eps10.0trainval-2024-10-20-09-39-26/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/celeba_male_32_eps10.0trainval-2024-10-22-08-00-43/gen/gen.npz",
        "/DPImageBench/exp/dp-kernel/camelyon_32_eps10.0trainval-2024-10-20-09-41-01/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/pe/mnist_28_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-08-24-53/gen/gen.npz",
        "/DPImageBench/exp/pe/fmnist_28_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-17-19-07/gen/gen.npz",
        "/DPImageBench/exp/pe/cifar10_32_eps10.0_trainval_th2_tr100_vds0246810-2024-10-30-17-42-57/gen/gen.npz",
        "/DPImageBench/exp/pe/cifar100_32_eps10.0_trainval_th2_tr100_vds0246810-2024-10-31-04-37-59/gen/gen.npz",
        "/DPImageBench/exp/pe/eurosat_32_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-17-33-12/gen/gen.npz",
        "/DPImageBench/exp/pe/celeba_male_32_eps10.0_trainval_th2_tr100_vds0246810-2024-11-02-17-24-12/gen/gen.npz",
        "/DPImageBench/exp/pe/camelyon_32_eps10.0_trainval_th4_tr10_vds00000111-2024-11-01-21-44-46/gen/gen.npz",
    ],
    [
        "/DPImageBench/exp/gs-wgan/mnist_28_eps10.0-2024-11-03-08-33-03/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/fmnist_28_eps10.0-2024-11-02-01-59-08/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/cifar10_32_eps10.0trainval-2024-11-04-22-47-20/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/cifar100_32_eps10.0trainval-2024-11-05-07-04-09/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/eurosat_32_eps10.0trainval-2024-11-04-00-31-57/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/celeba_male_32_eps10.0trainval_step5w-2024-11-04-21-58-52/gen/gen.npz",
        "/DPImageBench/exp/gs-wgan/camelyon_32_eps10.0trainval-2024-11-04-21-49-24/gen/gen.npz",
    ],
    [
        "/DPImageBench/exp/dpgan/mnist_28_eps10.0trainval-2024-10-22-11-42-58/gen/gen.npz",
        "/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval-2024-10-22-11-44-30/gen/gen.npz",
        "/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval-2024-10-22-21-58-29/gen/gen.npz",
        "/DPImageBench/exp/dpgan/cifar100_32_eps10.0trainval-2024-10-22-21-35-25/gen/gen.npz",
        "/DPImageBench/exp/dpgan/eurosat_32_eps10.0trainval-2024-10-22-12-46-17/gen/gen.npz",
        "/DPImageBench/exp/dpgan/celeba_male_32_eps10.0trainval-2024-10-22-14-42-52/gen/gen.npz",
        "/DPImageBench/exp/dpgan/camelyon_32_eps10.0trainval-2024-10-22-22-13-03/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dpdm/mnist_28_eps10.0trainval-2024-10-22-22-42-07/gen/gen.npz",
        "/DPImageBench/exp/dpdm/fmnist_28_eps10.0trainval-2024-10-23-19-31-12/gen/gen.npz",
        "/DPImageBench/exp/dpdm/cifar10_32_eps10.0trainval-2024-10-24-01-44-41/gen/gen.npz",
        "/DPImageBench/exp/dpdm/cifar100_32_eps10.0trainval-2024-10-26-01-37-36/gen/gen.npz",
        "/DPImageBench/exp/dpdm/eurosat_32_eps10.0trainval-2024-10-24-12-56-31/gen/gen.npz",
        "/DPImageBench/exp/dpdm/celeba_male_32_eps10.0trainval-2024-10-24-00-28-59/gen/gen.npz",
        "/DPImageBench/exp/dpdm/camelyon_32_eps10.0trainval-2024-10-25-02-30-03/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dp-feta/mnist_28_eps10.0val_central_mean-2025-03-19-07-56-07/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/fmnist_28_eps10.0val_central_mean-2025-03-20-06-25-53/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/cifar10_32_eps10.0val_central_mean-2025-03-23-07-47-44/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/cifar100_32_eps10.0val_central_mean-2025-03-22-08-43-07/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/eurosat_32_eps10.0val_central_mean-2025-03-22-11-11-55/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/celeba_male_32_eps10.0sen_central_mean-2025-03-19-04-42-04/gen/gen.npz",
        "/DPImageBench/exp/dp-feta/camelyon_32_eps10.0sen_central_mean-2025-03-19-04-42-04/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/pdp-diffusion/mnist_28_eps10.0val_cn1e-3-2024-11-29-07-04-47/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/fmnist_28_eps10.0val_cn1e-3-2024-12-02-00-23-19/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0val_cn1e-3-2024-12-04-07-16-42/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/cifar100_32_eps10.0val_cn1e-3-2024-11-29-13-25-40/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/eurosat_32_eps10.0val_cn1e-3-2024-12-02-20-33-19/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/celeba_male_32_eps10.0val_cn1e-3-2024-12-05-11-09-18/gen/gen.npz",
        "/DPImageBench/exp/pdp-diffusion/camelyon_32_eps10.0val_cn1e-3-2024-12-04-15-36-20/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dp-ldm-sd/mnist_28_eps10.0val_cn1e-3-2024-11-30-03-36-36/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/fmnist_28_eps10.0val_cn1e-3-2024-12-01-09-32-57/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/cifar10_32_eps10.0val_cn1e-3-2024-11-30-05-18-26/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/cifar100_32_eps10.0val_cn1e-3-2024-11-30-19-55-20/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/eurosat_32_eps10.0val_cn1e-3-2024-11-30-21-20-07/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/celeba_male_32_eps10.0val_cn1e-3-2024-12-02-18-57-56/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm-sd/camelyon_32_eps10.0sen_cn1e-3-2024-12-01-10-24-18/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dp-ldm/mnist_28_eps10.0val_large28-2025-01-13-02-29-08/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/fmnist_28_eps10.0val_large28-2025-01-14-11-53-18/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/cifar10_32_eps10.0val-2024-12-24-06-43-33/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/cifar100_32_eps10.0val-2024-12-25-07-54-55/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/eurosat_32_eps10.0val-2024-12-25-07-56-13/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/celeba_male_32_eps10.0val-2024-12-27-08-36-53/gen/gen.npz",
        "/DPImageBench/exp/dp-ldm/camelyon_32_eps10.0val-2024-12-28-03-09-09/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/dp-lora/mnist_28_eps10.0val_large28-2025-01-13-02-29-08/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/fmnist_28_eps10.0val_large28-2025-01-14-11-53-01/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/cifar10_32_eps10.0val-2024-12-27-22-36-33/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/cifar100_32_eps10.0val-2024-12-30-05-05-15/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/eurosat_32_eps10.0val-2024-12-29-07-34-37/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/celeba_male_32_eps10.0val-2024-12-30-04-11-56/gen/gen.npz",
        "/DPImageBench/exp/dp-lora/camelyon_32_eps10.0val-2024-12-30-03-19-40/gen/gen.npz"
    ],
    [
        "/DPImageBench/exp/privimage/mnist_28_eps10.0val_cn1e-3-2024-12-04-11-40-13/gen/gen.npz",
        "/DPImageBench/exp/privimage/fmnist_28_eps10.0val_cn1e-3-2024-12-05-05-29-20/gen/gen.npz",
        "/DPImageBench/exp/privimage/cifar10_32_eps10.0val_cn1e-3-2024-12-03-00-13-54/gen/gen.npz",
        "/DPImageBench/exp/privimage/cifar100_32_eps10.0val_cn1e-3-2024-12-10-22-41-35/gen/gen.npz",
        "/DPImageBench/exp/privimage/eurosat_32_eps10.0val_pre3200-2024-12-05-07-18-11/gen/gen.npz",
        "/DPImageBench/exp/privimage/celeba_male_32_eps10.0val_pre3200-2024-12-05-22-26-39/gen/gen.npz",
        "/DPImageBench/exp/privimage/camelyon_32_eps10.0val_pre3200-2024-12-05-22-30-44/gen/gen.npz"
    ],
    [
        "./dataset/mnist/train_28.zip",
        "./dataset/fmnist/train_28.zip",
        "./dataset/cifar10/train_32.zip",
        "./dataset/cifar100/train_32.zip",
        "./dataset/eurosat/train_32.zip",
        "./dataset/celeba/train_32_Male.zip",
        "./dataset/camelyon/train_32.zip",
    ]]

    for method_idx in range(len(methods)):
        for dataset_idx in range(len(datasets)):
            if method_idx == 0:
                axs[method_idx, dataset_idx].set_title(datasets[dataset_idx])
            if dataset_idx == 0:
                axs[method_idx, dataset_idx].set_ylabel(methods[method_idx], rotation=0, fontsize=9, horizontalalignment='left')
                axs[method_idx, dataset_idx].yaxis.set_label_coords(-0.63, 0.3)
            
            axs[method_idx, dataset_idx].set_xticks([])
            axs[method_idx, dataset_idx].set_yticks([])

            gen_path = gen_lists[method_idx][dataset_idx]
            if not os.path.exists(gen_path):
                continue

            if methods[method_idx] == "Real":
                syn_data, syn_labels = sample_real_images(gen_path)
            else:
                syn = np.load(gen_path)
                syn_data, syn_labels = syn["x"], syn["y"]
            num_classes = len(set(list(syn_labels)))

            img_patch = []
            if num_classes == 2:
                for cls in range(2):
                    cls_img = syn_data[syn_labels==cls]
                    cls_img = cls_img[:column_per_dataset//2].transpose(1, 2, 0, 3).reshape(cls_img.shape[1], cls_img.shape[2], cls_img.shape[3] * column_per_dataset//2)
                    img_patch.append(cls_img)
                img_patch = np.concatenate(img_patch, axis=2)
            else:
                for cls in range(column_per_dataset):
                    cls_img = syn_data[syn_labels==cls]
                    cls_img = cls_img[:row_per_method].transpose(1, 0, 2, 3).reshape(cls_img.shape[1], cls_img.shape[2] * row_per_method, cls_img.shape[3])
                    img_patch.append(cls_img)
                img_patch = np.concatenate(img_patch, axis=2)
            img_patch = (img_patch * 255.).astype('uint8').transpose(1, 2, 0)
            if img_patch.shape[-1] == 1:
                img_patch = np.concatenate([img_patch]*3, axis=-1)
            
            img_patch = Image.fromarray(img_patch)
            img_patch = img_patch.resize((width_per_patch, height_per_patch))

            axs[method_idx, dataset_idx].imshow(img_patch)
            axs[method_idx, dataset_idx].imshow(img_patch)
        

    fig.subplots_adjust(wspace=0.025, hspace=0.0)
    fig.savefig("eps10_visual.png", bbox_inches='tight')
    fig.savefig("eps10_visual.pdf", bbox_inches='tight')

visualize_main()