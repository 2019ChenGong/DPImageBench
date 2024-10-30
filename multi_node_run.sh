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
# CUDA_VISIBLE_DEVICES=2 python run.py -m GS-WGAN -dn celeba_male_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=4 python run.py -m GS-WGAN -dn celeba_male_32 -e 1.0 &

CUDA_VISIBLE_DEVICES=0,1 python eval.py -m PrivImage -dn eurosat_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/eurosat_32_eps1.0-2024-10-28-15-06-50 &
CUDA_VISIBLE_DEVICES=0,1 python eval.py sensitive_data.trainnum=val -m PrivImage -dn eurosat_32 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/eurosat_32_eps10.0trainval-2024-10-28-15-06-50 &
CUDA_VISIBLE_DEVICES=0,1 python eval.py sensitive_data.trainnum=val -m PrivImage -dn eurosat_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/privimage/eurosat_32_eps1.0trainval-2024-10-28-15-06-50 &
# python eval.py sensitive_data.train_num=val -m DP-LDM -dn cifar100_32 -e 1.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-ldm/cifar100_32_eps1.0unconditional_trainval-2024-10-26-00-55-37 &
# python eval.py sensitive_data.train_num=val -m DP-LDM -dn cifar100_32 -e 10.0 -ep /p/fzv6enresearch/DPImageBench/exp/dp-ldm/cifar100_32_eps10.0unconditional_trainval-2024-10-26-00-06-28 &

# python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/dp-ldm/cifar10_32_eps10.0unconditional_ch1224_nf64-2024-10-27-03-33-17/pretrain/checkpoints/final_checkpoint.pth model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=128 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_ch1224_nf64

# python run.py sensitive_data.train_num=val -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval