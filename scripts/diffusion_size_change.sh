# DPDM

# 140G 11.1M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=64 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=128 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=256 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=3 public_data.name=null sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=512 -m DPDM -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128

# PDP-Diffusion

# 140G 11.1M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=100 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=200 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128

# PrivImage
# 140G 11.1M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8] model.network.ch_mult=[1,2,3] model.network.nf=64 train.dp.n_splits=100 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch123_nf64

# 120G 19.6M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=200 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf64

# 120G 44.2M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=96 train.dp.n_splits=300 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf96

# 120G 78.5M
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=128 train.dp.n_splits=600 -m PrivImage -dn cifar10_32 -e 10.0 -ed trainval_ch1224_nf128


# DP-LDM
# effects of pretrain_epochs, 
# 75G
python run.py setup.n_gpus_per_node=3 sensitive_data.train_num=val pretrain.n_epochs=50 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=64 -m DP-LDM -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1224_nf64_pre50

# 75G
python run.py setup.n_gpus_per_node=3 sensitive_data.train_num=val pretrain.n_epochs=10 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,2] model.network.nf=128 train.dp.n_splits=128 -m DP-LDM -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1222_nf128

# 99G
python run.py setup.n_gpus_per_node=3 sensitive_data.train_num=val pretrain.n_epochs=10 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,1,2,2] model.network.nf=192 train.dp.n_splits=128 -m DP-LDM -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1222_nf192


# PDP-Diffusion
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val pretrain.n_epochs=10 model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.dp.n_splits=64 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1224_nf64
# finetune
python run.py setup.n_gpus_per_node=3 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0unconditional_trainval_ch1224_nf64_pre10-2024-10-29-19-01-45/pretrain/checkpoints/final_checkpoint.pth sensitive_data.train_num=val model.network.attn_resolutions=[16] model.network.attn_resolutions=[16,8,4] model.network.ch_mult=[1,2,2,4] model.network.nf=64 train.n_epochs=80 train.batch_size=16384 train.dp.max_grad_norm=0.001 train.dp.n_splits=512 train.loss.label_unconditioning_prob=0.0 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_ftnodrop_trainval_ch1224_nf64_pre10


# pretrain
python run.py setup.n_gpus_per_node=3 sensitive_data.train_num=val pretrain.n_epochs=10 model.network.attn_resolutions=[16] model.network.ch_mult=[1,2,2,2] model.network.nf=128 train.dp.n_splits=64 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1222_nf128
# finetune
python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0unconditional_trainval_ch1222_nf128-2024-10-29-08-09-16/pretrain/checkpoints/final_checkpoint.pth sensitive_data.train_num=val model.network.attn_resolutions=[16] model.network.ch_mult=[1,2,2,2] model.network.nf=128 train.n_epochs=80 train.batch_size=16384 train.dp.max_grad_norm=0.001 train.dp.n_splits=600 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_ftnodrop_trainval_ch1222_nf128

# pretrain
python run.py setup.n_gpus_per_node=4 sensitive_data.train_num=val pretrain.n_epochs=10 model.network.attn_resolutions=[16] model.network.ch_mult=[1,1,2,2] model.network.nf=192 train.dp.n_splits=64 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1222_nf192
# finetune
python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/cifar10_32_eps10.0unconditional_trainval_ch1222_nf128-2024-10-29-08-09-16/pretrain/checkpoints/final_checkpoint.pth sensitive_data.train_num=val model.network.attn_resolutions=[16] model.network.ch_mult=[1,2,2,2] model.network.nf=192 train.n_epochs=80 train.batch_size=16384 train.dp.max_grad_norm=0.001 train.dp.n_splits=512 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_trainval_ch1222_nf192