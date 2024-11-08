#!/bin/bash 
# master_address='128.143.63.115'
# n_nodes=4
# p=gnolim
# node=ai

# srun -w ${node}'01' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=0 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
# sleep 20
# srun -w ${node}'02' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=1 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
# srun -w ${node}'03' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=2 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
# srun -w ${node}'04' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=3 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
#srun -w ai04 -p gnolim python main.py --mode train setup.node_rank=2 &
#srun -w ai05 -p gnolim python main.py --mode train setup.node_rank=3 &

# wait
# hostname -I 查看addr
# jaguar03 128.143.136.9
# lynx01 128.143.69.60
# ai01 128.143.63.115
# ai04 128.143.63.198
# affogato12 128.143.69.81
# affogato11 128.143.69.75


# CUDA_VISIBLE_DEVICES=1 python run.py -m GS-WGAN -dn eurosat_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=2 python run.py -m GS-WGAN -dn eurosat_32 -e 1.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=60 model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar10/pretrain/BigGAN_3.8M_trainval -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_3.8M &
# CUDA_VISIBLE_DEVICES=2 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=80 model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar10/pretrain/BigGAN_6.6M_trainval -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_6.6M &

# to eval
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DP-Kernel -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-kernel/cifar10_32_eps10.0trainval_eps20-2024-10-29-05-30-19 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DP-MERF -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-merf/cifar10_32_eps10.0trainval_eps15-2024-10-29-00-07-10 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DPDM -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpdm/fmnist_28_eps1.0trainval_eps0.2-2024-10-29-01-51-05 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DPDM -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpdm/fmnist_28_eps10.0trainval_eps15-2024-10-28-16-18-48 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DPDM -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpdm/cifar10_32_eps10.0trainval_eps5-2024-10-28-04-31-08 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DPGAN -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval_eps5-2024-10-29-04-29-58 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m DPGAN -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval_eps15-2024-10-29-04-35-12 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m PrivImage -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/cifar10_32_eps10.0trainval_eps5-2024-10-29-00-02-50 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m PrivImage -dn cifar10_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/cifar10_32_eps10.0trainval_eps20-2024-10-29-00-57-50 &
# python eval.py -m PE -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps5-2024-10-29-00-02-50 &
# python eval.py -m PE -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps20-2024-10-29-00-57-21 &


# gan model size
# CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=140 gen.batch_size=200 -m DP-Kernel -dn cifar10_32 -e 10.0 -ed trainval_19.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=120 gen.batch_size=200 -m DP-Kernel -dn cifar10_32 -e 10.0 -ed trainval_14.3M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=140 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_19.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null train.n_splits=2 sensitive_data.train_num=val model.Generator.g_conv_dim=120 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_19.4M &
# CUDA_VISIBLE_DEVICES=0,1,2 python run.py setup.n_gpus_per_node=4 public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=120 -m DPGAN -dn cifar10_32 -e 10.0 -ed trainval_19.4M &
# CUDA_VISIBLE_DEVICES=7 python run.py public_data.name=null sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 10.0 -ed trainval_1.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 10.0 -ed trainval_1.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 10.0 -ed trainval_1.4M &


# CUDA_VISIBLE_DEVICES=0 python eval.py sensitive_data.train_num=val -m PE -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps15_LZN-2024-11-06-23-49-40  &
# CUDA_VISIBLE_DEVICES=1 python eval.py sensitive_data.train_num=val -m PE -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps20_LZN-2024-11-06-23-49-58  &

# CUDA_VISIBLE_DEVICES=4 python eval.py sensitive_data.train_num=val -m PE -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pe/fmnist_28_eps0.2_trainval_th44_tr10_vds001-2024-11-03-22-47-07  &
# CUDA_VISIBLE_DEVICES=5 python eval.py sensitive_data.train_num=val -m PE -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pe/fmnist_28_eps5.0_trainval_th5_tr10_vds000011112222-2024-11-03-22-43-48  &
# CUDA_VISIBLE_DEVICES=6 python eval.py sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps0.2_LZN-2024-11-06-23-49-45  &
# CUDA_VISIBLE_DEVICES=7 python eval.py sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps5_LZN-2024-11-06-23-50-04  &
# CUDA_VISIBLE_DEVICES=0 python eval.py sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps15_LZN-2024-11-06-23-49-40  &
CUDA_VISIBLE_DEVICES=0 python eval.py sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0 -ep //p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/fmnist_28_eps1.0_trainval_eps20_LZN-2024-11-06-23-49-58  &
# python eval.py sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps5-2024-10-29-00-02-50 &
# python eval.py sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps15-2024-10-29-00-02-50 &
# python eval.py sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps20-2024-10-29-00-57-21 &
# python eval.py sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_1.4M-2024-11-05-06-50-13 &
# python eval.py sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-kernel/cifar10_32_eps10.0trainval_3.8M-2024-11-05-02-03-07 &
# python eval.py sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval_1.4M-2024-11-05-02-44-13 &

# python run.py sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 10.0 -ed trainval

# CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar10/pretrain/BigGAN_3.8M_trainval model.Generator.g_conv_dim=60 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_3.8M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar10/pretrain/BigGAN_6.6M_trainval model.Generator.g_conv_dim=80 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_6.6M &
