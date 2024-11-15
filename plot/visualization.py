import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from data.stylegan3.dataset import ImageFolderDataset


fontsize = 8
# matplotlib.rcParams.update({'font.size': fontsize, 'font.family': 'Arial', 'font.weight': 'normal'})
matplotlib.rcParams.update({'font.size': fontsize, 'font.weight': 'normal'})


def sample_real_images(data_path):
    dataset = ImageFolderDataset(data_path, use_labels=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=2000)
    for x, y in dataloader:
        x = x.to(torch.float32) / 255.
        y = torch.argmax(y, dim=1)
        break
    x = x.numpy()
    y = y.numpy()
    return x, y


def visualize_app_1():
    column_per_dataset = 10
    row_per_method = 2
    img_size = 32
    datasets = ["MNIST", "F-MNIST", "CIFAR-10", "EuroSAT"]
    methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM", "PrivImage"]
    width_per_patch = img_size * column_per_dataset
    height_per_patch = img_size * row_per_method

    fig = plt.figure(figsize=(4.5, 3))
    axs = fig.subplots(len(methods), len(datasets))

    gen_lists = [
    [
        "/exp/dp-merf/mnist_28_eps1.0-2024-10-06-23-27-19",
        "/exp/dp-merf/fmnist_28_eps1.0-2024-10-06-23-27-19",
        "/exp/dp-merf/cifar10_32_eps1.0-2024-10-07-16-20-58",
        "/exp/dp-merf/eurosat_32_eps1.0-2024-10-07-13-23-29",
    ],  
    [
        "/exp/dp-ntk/mnist_28_eps1.0-2024-10-07-00-40-41",
        "/exp/dp-ntk/fmnist_28_eps1.0-2024-10-07-00-43-41",
        "/exp/dp-ntk/cifar10_32_eps1.0-2024-10-07-16-22-49",
        "/exp/dp-ntk/eurosat_32_eps1.0-2024-10-07-00-46-23",
    ],
    [
        "/exp/dp-kernel/mnist_28_eps1.0-2024-10-06-23-53-17",
        "/exp/dp-kernel/fmnist_28_eps1.0-2024-10-06-23-53-17",
        "/exp/dp-kernel/cifar10_32_eps1.0-2024-10-07-16-22-49",
        "/exp/dp-kernel/eurosat_32_eps1.0-2024-10-06-23-53-17",
    ],
    [
        "",
        "",
        "",
        "",
    ],
    [
        "/exp/dpgan/mnist_28_eps1.0-2024-10-07-23-48-04",
        "/exp/dpgan/fmnist_28_eps1.0-2024-10-08-00-08-03",
        "/exp/dpgan/cifar10_32_eps1.0-2024-10-08-14-34-51",
        "/exp/dpgan/eurosat_32_eps1.0-2024-10-08-01-15-27",
    ],
    [
        "/exp/dpdm/mnist_28_eps1.0-2024-10-08-23-37-39",
        "/exp/dpdm/fmnist_28_eps1.0-2024-10-09-14-42-01",
        "/exp/dpdm/cifar10_32_eps1.0-2024-10-08-15-55-27",
        "/exp/dpdm/eurosat_32_eps1.0-2024-10-09-21-06-55",
    ],
    [
        "/exp/pdp-diffusion/mnist_28_eps1.0_LZN-2024-10-25-23-09-18",
        "/exp/pdp-diffusion/fmnist_28_eps1.0_LZN-2024-10-26-03-27-14",
        "/exp/pdp-diffusion/cifar10_32_eps1.0_LZN-2024-10-26-03-26-48",
        "/exp/pdp-diffusion/eurosat_32_eps1.0_LZN-2024-10-27-19-53-32",
    ],
    [
        "/exp/dp-ldm/mnist_28_eps1.0-2024-10-19-00-08-31",
        "/exp/dp-ldm/fmnist_28_eps1.0-2024-10-19-16-56-37",
        "/exp/dp-ldm/cifar10_32_eps1.0-2024-10-19-22-37-57",
        "/exp/dp-ldm/eurosat_32_eps1.0-2024-10-20-16-33-13",
    ],
    [
        "/exp/privimage/mnist_28_eps1.0-2024-10-09-05-30-15",
        "/exp/privimage/fmnist_28_eps1.0-2024-10-16-04-32-21",
        "/exp/privimage/cifar10_32_eps1.0-2024-10-16-07-21-01",
        "/exp/privimage/eurosat_32_eps1.0-2024-10-28-15-06-50",
    ]]

    for method_idx in range(len(methods)):
        for dataset_idx in range(len(datasets)):
            if method_idx == 0:
                axs[method_idx, dataset_idx].set_title(datasets[dataset_idx])
            if dataset_idx == 0:
                x = [-0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36]
                axs[method_idx, dataset_idx].set_ylabel(methods[method_idx], rotation=0, fontsize=9)
                axs[method_idx, dataset_idx].yaxis.set_label_coords(x[method_idx], 0.35)
            
            axs[method_idx, dataset_idx].set_xticks([])
            axs[method_idx, dataset_idx].set_yticks([])


            # gen_path = os.path.join(gen_lists[method_idx][dataset_idx], "gen", "gen.npz")
            # if not os.path.exists(gen_path):
            #     continue
            # syn = np.load(gen_path)
            # syn_data, syn_labels = syn["x"], syn["y"]
            # num_classes = len(set(list(syn_labels)))

            # img_patch = []
            # if num_classes == 2:
            #     for cls in range(2):
            #         cls_img = syn_data[syn_labels==cls]
            #         cls_img = cls_img[:column_per_dataset].transpose(1, 2, 0, 3).reshape(cls_img.shape[1], cls_img.shape[2], cls_img.shape[3] * column_per_dataset)
            #         img_patch.append(cls_img)
            #     img_patch = np.concatenate(img_patch, axis=1)
            # else:
            #     for cls in range(column_per_dataset):
            #         cls_img = syn_data[syn_labels==cls]
            #         cls_img = cls_img[:row_per_method].transpose(1, 0, 2, 3).reshape(cls_img.shape[1], cls_img.shape[2] * row_per_method, cls_img.shape[3])
            #         img_patch.append(cls_img)
            #     img_patch = np.concatenate(img_patch, axis=2)
            # img_patch = (img_patch * 255.).astype('uint8').transpose(1, 2, 0)
            # if img_patch.shape[-1] == 1:
            #     img_patch = np.concatenate([img_patch]*3, axis=-1)
            
            img_patch = (np.random.rand(32*row_per_method, 32*column_per_dataset) * 255.).astype('uint8')
            img_patch = Image.fromarray(img_patch)
            img_patch = img_patch.resize((width_per_patch, height_per_patch))

            axs[method_idx, dataset_idx].imshow(img_patch)
        
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig("eps1_visual_1.png", bbox_inches='tight')
    fig.savefig("eps1_visual_1.pdf", bbox_inches='tight')

def visualize_app_2():
    column_per_dataset_list = [10, 10, 20]
    row_per_method = 2
    img_size = 32
    height_per_patch = img_size * row_per_method
    datasets = ["CelebA", "Camelyon", "CIFAR-100"]
    methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM", "PrivImage"]

    fig = plt.figure(figsize=(11.5, 5.2))

    gen_lists = [
    [
        "/exp/dp-merf/mnist_28_eps1.0-2024-10-06-23-27-19",
        "/exp/dp-merf/fmnist_28_eps1.0-2024-10-06-23-27-19",
        "/exp/dp-merf/cifar100_32_eps1.0-2024-10-07-16-20-58",
    ],  
    [
        "/exp/dp-ntk/celeba_male_32_eps1.0-2024-10-07-20-00-33",
        "/exp/dp-ntk/celeba_male_32_eps1.0-2024-10-07-20-00-33",
        "/exp/dp-ntk/camelyon_32_eps1.0-2024-10-07-00-46-23",
    ],
    [
        "/exp/dp-kernel/celeba_male_32_eps1.0-2024-10-06-23-53-17",
        "/exp/dp-kernel/camelyon_32_eps1.0-2024-10-06-23-55-05",
        "/exp/dp-kernel/cifar100_32_eps1.0-2024-10-09-15-12-05",
    ],
    [
        "",
        "",
        "",
        "",
    ],
    [
        "/exp/dpgan/celeba_male_32_eps1.0-2024-10-08-12-07-22",
        "/exp/dpgan/camelyon_32_eps1.0-2024-10-08-12-11-42",
        "/exp/dpgan/cifar10_32_eps1.0-2024-10-08-14-34-51",
    ],
    [
        "/exp/dpdm/celeba_male_32_eps1.0-2024-10-11-23-12-23",
        "/exp/dpdm/camelyon_32_eps1.0-2024-10-10-17-56-23",
        "/exp/dpdm/cifar100_32_eps1.0-2024-10-13-10-24-41",
    ],
    [
        "/exp/pdp-diffusion/celeba_male_32_eps1.0_LZN-2024-10-28-05-48-53",
        "/exp/pdp-diffusion/camelyon_32_eps1.0_LZN-2024-10-28-21-49-21",
        "/exp/pdp-diffusion/cifar10_32_eps1.0_LZN-2024-10-26-03-26-48",
    ],
    [
        "/exp/dp-ldm/celeba_male_32_eps1.0unconditional-2024-10-25-20-13-02",
        "/exp/dp-ldm/camelyon_32_eps1.0unconditional-2024-10-25-10-58-19",
        "/exp/dp-ldm/cifar100_32_eps1.0-2024-10-20-15-09-59",
    ],
    [
        "/exp/privimage/celeba_male_32_eps1.0-2024-10-17-02-16-40",
        "/exp/privimage/camelyon_32_eps1.0-2024-10-18-07-02-27",
        "/exp/privimage/cifar100_32_eps1.0-2024-10-17-00-27-34",
    ]]

    for method_idx in range(len(methods)):
        for dataset_idx in range(len(datasets)):
            if dataset_idx == 2:
                ax = fig.add_subplot(len(methods), 2, (method_idx+1)*2)
            else:
                ax = fig.add_subplot(len(methods), 4, method_idx*4+dataset_idx+1)
            column_per_dataset = column_per_dataset_list[dataset_idx]
            width_per_patch = img_size * column_per_dataset
            if method_idx == 0:
                ax.set_title(datasets[dataset_idx])
            if dataset_idx == 0:
                x = [-0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36, -0.36]
                ax.set_ylabel(methods[method_idx], rotation=0, fontsize=9)
                ax.yaxis.set_label_coords(x[method_idx], 0.35)
            
            ax.set_xticks([])
            ax.set_yticks([])


            # gen_path = os.path.join(gen_lists[method_idx][dataset_idx], "gen", "gen.npz")
            # if not os.path.exists(gen_path):
            #     continue
            # syn = np.load(gen_path)
            # syn_data, syn_labels = syn["x"], syn["y"]
            # num_classes = len(set(list(syn_labels)))

            # img_patch = []
            # if num_classes == 2:
            #     for cls in range(2):
            #         cls_img = syn_data[syn_labels==cls]
            #         cls_img = cls_img[:column_per_dataset].transpose(1, 2, 0, 3).reshape(cls_img.shape[1], cls_img.shape[2], cls_img.shape[3] * column_per_dataset)
            #         img_patch.append(cls_img)
            #     img_patch = np.concatenate(img_patch, axis=1)
            # else:
            #     for cls in range(column_per_dataset):
            #         cls_img = syn_data[syn_labels==cls]
            #         cls_img = cls_img[:row_per_method].transpose(1, 0, 2, 3).reshape(cls_img.shape[1], cls_img.shape[2] * row_per_method, cls_img.shape[3])
            #         img_patch.append(cls_img)
            #     img_patch = np.concatenate(img_patch, axis=2)
            # img_patch = (img_patch * 255.).astype('uint8').transpose(1, 2, 0)
            # if img_patch.shape[-1] == 1:
            #     img_patch = np.concatenate([img_patch]*3, axis=-1)
            
            img_patch = (np.random.rand(32*row_per_method, 32*column_per_dataset) * 255.).astype('uint8')
            img_patch = Image.fromarray(img_patch)
            img_patch = img_patch.resize((width_per_patch, height_per_patch))

            ax.imshow(img_patch)
        
    fig.subplots_adjust(wspace=0.1, hspace=-0.1)
    fig.savefig("eps1_visual_2.png", bbox_inches='tight')
    fig.savefig("eps1_visual_2.pdf", bbox_inches='tight')
    

def visualize_main():
    column_per_dataset = 6
    row_per_method = 1
    img_size = 32
    datasets = ["MNIST", "F-MNIST", "CIFAR-10", "CIFAR-100", "EuroSAT", "CelebA", "Camelyon"]
    methods = ["DP-MERF", "DP-NTK", "DP-Kernel", "PE", "GS-WGAN", "DP-GAN", "DPDM", "PDP-Diffusion", "DP-LDM", "PrivImage", "Real"]
    width_per_patch = img_size * column_per_dataset
    height_per_patch = img_size * row_per_method

    fig = plt.figure(figsize=(13.6, 3.8))
    axs = fig.subplots(len(methods), len(datasets))

    gen_lists = [
    [
        "/exp/dp-merf/mnist_28_eps10.0-2024-10-06-23-27-19/gen/gen.npz",
        "/exp/dp-merf/fmnist_28_eps10.0-2024-10-06-23-27-19/gen/gen.npz",
        "/exp/dp-merf/cifar10_32_eps10.0-2024-10-07-16-20-58/gen/gen.npz",
        "/exp/dp-merf/cifar100_32_eps10.0-2024-10-07-16-20-58/gen/gen.npz",
        "/exp/dp-merf/eurosat_32_eps10.0-2024-10-07-13-23-29/gen/gen.npz",
        "/exp/dp-merf/celeba_male_32_eps10.0-2024-10-07-16-17-22/gen/gen.npz",
        "/exp/dp-merf/camelyon_32_eps10.0-2024-10-07-20-32-22/gen/gen.npz"
    ],  
    [
        "/exp/dp-ntk/mnist_28_eps10.0-2024-10-07-00-45-38/gen/gen.npz",
        "/exp/dp-ntk/fmnist_28_eps10.0-2024-10-07-00-43-41/gen/gen.npz",
        "/exp/dp-ntk/cifar10_32_eps10.0-2024-10-07-16-22-49/gen/gen.npz",
        "/exp/dp-ntk/cifar100_32_eps10.0-2024-10-07-16-22-49/gen/gen.npz",
        "/exp/dp-ntk/eurosat_32_eps10.0-2024-10-07-00-46-23/gen/gen.npz",
        "/exp/dp-ntk/celeba_male_32_eps10.0-2024-10-07-20-00-33/gen/gen.npz",
        "/exp/dp-ntk/camelyon_32_eps10.0-2024-10-07-00-46-23/gen/gen.npz"
    ],
    [
        "/exp/dp-kernel/mnist_28_eps10.0-2024-10-06-23-53-17/gen/gen.npz",
        "/exp/dp-kernel/fmnist_28_eps10.0-2024-10-06-23-53-17/gen/gen.npz",
        "/exp/dp-kernel/cifar10_32_eps10.0-2024-10-07-16-22-49/gen/gen.npz",
        "/exp/dp-kernel/cifar100_32_eps10.0-2024-10-08-14-59-38/gen/gen.npz",
        "/exp/dp-kernel/eurosat_32_eps10.0-2024-10-06-23-53-17/gen/gen.npz",
        "/exp/dp-kernel/celeba_male_32_eps10.0-2024-10-09-15-12-05/gen/gen.npz",
        "/exp/dp-kernel/camelyon_32_eps10.0-2024-10-06-23-55-05/gen/gen.npz"
    ],
    [
        "/exp/pe/mnist_28_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-08-24-53/gen/gen.npz",
        "/exp/pe/fmnist_28_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-17-19-07/gen/gen.npz",
        "/exp/pe/cifar10_32_eps10.0_trainval_th2_tr100_vds0246810-2024-10-30-17-42-57/gen/gen.npz",
        "/exp/pe/cifar100_32_eps10.0_trainval_th2_tr100_vds0246810-2024-10-31-04-37-59/gen/gen.npz",
        "/exp/pe/eurosat_32_eps10.0_trainval_th3_tr10_vds000011112222-2024-11-01-17-33-12/gen/gen.npz",
        "/exp/pe/celeba_male_32_eps10.0_th2_tr100_vds0246810-2024-11-01-05-10-39/gen/gen.npz",
        "/exp/pe/camelyon_32_eps10.0_trainval_th4_tr10_vds00000111-2024-11-01-21-44-46/gen/gen.npz",
    ],
    [
        "/exp/gs-wgan/mnist_28_eps10.0-2024-11-03-08-33-03/gen/gen.npz",
        "/exp/gs-wgan/fmnist_28_eps10.0-2024-11-02-01-59-08/gen/gen.npz",
        "/exp/gs-wgan/cifar10_32_eps10.0-2024-11-03-00-10-53/gen/gen.npz",
        "/exp/gs-wgan/cifar100_32_eps10.0-2024-11-03-08-23-12/gen/gen.npz",
        "/exp/gs-wgan/eurosat_32_eps10.0-2024-11-03-10-39-01/gen/gen.npz",
        "/exp/gs-wgan/celeba_male_32_eps10.0trainval_step5w-2024-11-04-21-58-52/gen/gen.npz",
        "/exp/gs-wgan/camelyon_32_eps10.0trainval-2024-11-04-21-49-24/gen/gen.npz",
    ],
    [
        "/exp/dpgan/mnist_28_eps10.0-2024-10-08-00-06-54/gen/gen.npz",
        "/exp/dpgan/fmnist_28_eps10.0-2024-10-08-12-03-44/gen/gen.npz",
        "/exp/dpgan/cifar10_32_eps10.0-2024-10-08-14-35-04/gen/gen.npz",
        "/exp/dpgan/cifar100_32_eps10.0-2024-10-08-23-54-16/gen/gen.npz",
        "/exp/dpgan/eurosat_32_eps10.0-2024-10-08-01-19-27/gen/gen.npz",
        "/exp/dpgan/celeba_male_32_eps10.0-2024-10-08-12-09-30/gen/gen.npz",
        "/exp/dpgan/camelyon_32_eps10.0-2024-10-08-15-17-32/gen/gen.npz"
    ],
    [
        "/exp/dpdm/mnist_28_eps10.0-2024-10-09-14-26-40/gen/gen.npz",
        "/exp/dpdm/fmnist_28_eps10.0-2024-10-10-11-42-34/gen/gen.npz",
        "/exp/dpdm/cifar10_32_eps10.0-2024-10-10-21-41-21/gen/gen.npz",
        "/exp/dpdm/cifar100_32_eps10.0-2024-10-13-10-12-50/gen/gen.npz",
        "/exp/dpdm/eurosat_32_eps10.0-2024-10-10-14-39-52/gen/gen.npz",
        "/exp/dpdm/celeba_male_32_eps10.0-2024-10-11-15-22-52/gen/gen.npz",
        "/exp/dpdm/camelyon_32_eps10.0-2024-10-10-20-39-43/gen/gen.npz"
    ],
    [
        "/exp/pdp-diffusion/mnist_28_eps10.0Unconditional-2024-10-23-05-45-33/gen/gen.npz",
        "/exp/pdp-diffusion/fmnist_28_eps10.0Unconditional-2024-10-23-08-46-07/gen/gen.npz",
        "/exp/pdp-diffusion/cifar10_32_eps10.0_LZN-2024-10-26-00-08-47/gen/gen.npz",
        "/exp/pdp-diffusion/cifar100_32_eps10.0_LZN-2024-10-26-03-53-56/gen/gen.npz",
        "/exp/pdp-diffusion/eurosat_32_eps10.0Unconditional-2024-10-21-23-34-18/gen/gen.npz",
        "/exp/pdp-diffusion/celeba_male_32_eps10.0_trainval_LZN-2024-10-28-05-49-50/gen/gen.npz",
        "/exp/pdp-diffusion/camelyon_32_eps1.0_LZN-2024-10-28-21-49-21/gen/gen.npz"
    ],
    [
        "/exp/dp-ldm/mnist_28_eps10.0unconditional-2024-10-24-09-51-58/gen/gen.npz",
        "/exp/dp-ldm/fmnist_28_eps10.0unconditional-2024-10-24-20-59-35/gen/gen.npz",
        "/exp/dp-ldm/cifar10_32_eps10.0unconditional-2024-10-24-22-18-33/gen/gen.npz",
        "/exp/dp-ldm/cifar100_32_eps10.0unconditional-2024-10-25-07-00-22/gen/gen.npz",
        "/exp/dp-ldm/eurosat_32_eps10.0unconditional-2024-10-25-10-01-21/gen/gen.npz",
        "/exp/dp-ldm/celeba_male_32_eps10.0unconditional-2024-10-25-20-11-00/gen/gen.npz",
        "/exp/dp-ldm/camelyon_32_eps10.0unconditional-2024-10-25-11-13-20/gen/gen.npz"
    ],
    [
        "/exp/privimage/mnist_28_eps10.0-2024-10-15-12-52-07/gen/gen.npz",
        "/exp/privimage/fmnist_28_eps10.0-2024-10-16-02-21-25/gen/gen.npz",
        "/exp/privimage/cifar10_32_eps10.0-2024-10-16-17-28-44/gen/gen.npz",
        "/exp/privimage/cifar100_32_eps10.0-2024-10-17-00-29-00/gen/gen.npz",
        "/exp/privimage/eurosat_32_eps10.0-2024-10-27-10-02-36/gen/gen.npz",
        "/exp/privimage/celeba_male_32_eps10.0-2024-10-17-01-49-13/gen/gen.npz",
        "/exp/privimage/camelyon_32_eps10.0-2024-10-18-07-00-29/gen/gen.npz"
    ],
    [
        "/dataset/mnist/train_28.zip",
        "/dataset/fmnist/train_28.zip",
        "/dataset/cifar10/train_32.zip",
        "/dataset/cifar100/train_32.zip",
        "/dataset/eurosat/train_32.zip",
        "/dataset/celeba/train_32_Male.zip",
        "/dataset/camelyon/train_32.zip",
    ]]

    for method_idx in range(len(methods)):
        for dataset_idx in range(len(datasets)):
            if method_idx == 0:
                axs[method_idx, dataset_idx].set_title(datasets[dataset_idx])
            if dataset_idx == 0:
                axs[method_idx, dataset_idx].set_ylabel(methods[method_idx], rotation=0, fontsize=9, horizontalalignment='left')
                axs[method_idx, dataset_idx].yaxis.set_label_coords(-0.76, 0.35)
            
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


            # break
        

    fig.subplots_adjust(wspace=0.01, hspace=0.08)
    fig.savefig("eps10_visual.png", bbox_inches='tight')
    fig.savefig("eps10_visual.pdf", bbox_inches='tight')


visualize_main()
# visualize_app_1()
# visualize_app_2()