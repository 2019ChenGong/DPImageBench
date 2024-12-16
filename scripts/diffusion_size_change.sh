# DPDM

# 140G 11.1M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=64 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=128 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=256 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=4 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=512 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128

# PDP-Diffusion

# 140G 11.1M
python run.py setup.n_gpus_per_node=8 eval.mode=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=100 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed pretrain_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=4 eval.mode=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=200 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed pretrain_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=4 eval.mode=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed pretrain_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=4 eval.mode=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed pretrain_ch1224_nf128

# 78.5M
# reproduce
python run.py setup.n_gpus_per_node=3 eval.mode=val pretrain.batch_size=1024 pretrain.n_epochs=160 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128_repro

python run.py setup.n_gpus_per_node=4 eval.mode=val public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0trainval_ch1224_nf128_repro-2024-11-22-23-31-27/pretrain/checkpoints/final_checkpoint.pth train.n_epochs=60 train.dp.max_grad_norm=0.001 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=64 -m DP-LDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128_repro

# PrivImage
# 140G 11.1M
python run.py setup.n_gpus_per_node=1 sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=100 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=1 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=200 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
CUDA_VISIBLE_DEVICES=2,3 python run.py setup.n_gpus_per_node=2 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
CUDA_VISIBLE_DEVICES=2,3 python run.py setup.n_gpus_per_node=2 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128


# DP-LDM
# 140G 11.1M
python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=100 -m DP-LDM -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=3 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=200 -m DP-LDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=4 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m DP-LDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=4 pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m DP-LDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128

#  eval

python run.py setup.n_gpus_per_node=4 public_data.name=null sensitive_data.name=null model.ckpt=/exp/pdp-diffusion/cifar10_32_eps10.0_trainval_ch1224_nf96_LZN-2024-11-09-18-05-14/train/checkpoints/final_checkpoint.pth model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.name=null model.ckpt=/exp/dpdm/cifar10_32_eps10.0trainval_ch1224_nf128-2024-11-09-10-51-53/train/checkpoints/snapshot_checkpoint.pth model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=300 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96