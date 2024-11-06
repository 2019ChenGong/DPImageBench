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
# CUDA_VISIBLE_DEVICES=3 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar100/pretrain/ResNet_default_trainval -m GS-WGAN -dn cifar100_32 -e 10.0 -ed trainval &
# CUDA_VISIBLE_DEVICES=2 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/p/fzv6enresearch/DPImageBench/GS-WGAN/results/cifar100/pretrain/ResNet_default_trainval -m GS-WGAN -dn cifar100_32 -e 1.0 -ed trainval &

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
# CUDA_VISIBLE_DEVICES=1 python eval.py -m PrivImage -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps5-2024-10-29-00-02-50 &
# CUDA_VISIBLE_DEVICES=1 python eval.py -m PrivImage -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/fmnist_28_eps10.0trainval_eps20-2024-10-29-00-57-21 &


# gan model size
# CUDA_VISIBLE_DEVICES=3 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DP-Kernel -dn cifar10_32 -e 10.0 -ed trainval_10M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_10M &
# CUDA_VISIBLE_DEVICES=3 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_10M &
# CUDA_VISIBLE_DEVICES=0,1,2 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DPGAN -dn cifar10_32 -e 10.0 -ed trainval_10M &
# CUDA_VISIBLE_DEVICES=7 python run.py public_data.name=null sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 10.0 -ed trainval_1.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 10.0 -ed trainval_1.4M &
# CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 10.0 -ed trainval_1.4M &


python eval.py sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps1.0trainval_eps0.2-2024-10-28-23-46-11 &
python eval.py sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps5-2024-10-29-04-05-26 &
python eval.py sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps15-2024-10-29-04-30-41 &
python eval.py sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps20-2024-10-29-04-35-29 &
# python eval.py sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_1.4M-2024-11-05-06-50-13 &
# python eval.py sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-kernel/cifar10_32_eps10.0trainval_3.8M-2024-11-05-02-03-07 &
# python eval.py sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval_1.4M-2024-11-05-02-44-13 &

# python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-ldm/cifar10_32_eps10.0unconditional_ch1224_nf64-2024-10-27-03-33-17/pretrain/checkpoints/final_checkpoint.pth model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=128 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_ch1224_nf64

# python run.py sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 10.0 -ed trainval

python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val train.dp.epsilon=0.2 -m PDP-Diffusion -dn fmnist_28 -e 1.0 -ed trainval_eps0.2
python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val train.dp.epsilon=5 -m PDP-Diffusion -dn fmnist_28 -e 10 -ed trainval_eps5
python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val train.dp.epsilon=15 -m PDP-Diffusion -dn fmnist_28 -e 10 -ed trainval_eps15
python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val train.dp.epsilon=20 -m PDP-Diffusion -dn fmnist_28 -e 10 -ed trainval_eps20