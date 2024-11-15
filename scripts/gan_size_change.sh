CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=60 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_3.8M &
CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=80 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_6.6M &
CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_10.0M &
CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=120 -m DP-MERF -dn cifar10_32 -e 10.0 -ed trainval_14.3M &


CUDA_VISIBLE_DEVICES=0 python run.py pretrain.n_splits=1 train.n_splits=1 public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=60 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_3.8M &
CUDA_VISIBLE_DEVICES=1 python run.py pretrain.n_splits=1 train.n_splits=1 public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=80 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_6.6M &
CUDA_VISIBLE_DEVICES=2 python run.py pretrain.n_splits=2 train.n_splits=2 public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=100 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_10.0M &
CUDA_VISIBLE_DEVICES=3 python run.py pretrain.n_splits=2 train.n_splits=2 public_data.name=null sensitive_data.train_num=val model.Generator.g_conv_dim=120 -m DP-NTK -dn cifar10_32 -e 10.0 -ed trainval_14.3M &


CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/results/cifar10/pretrain/BigGAN_3.8M_trainval model.Generator.g_conv_dim=60 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_3.8M &
CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/results/cifar10/pretrain/BigGAN_6.6M_trainval model.Generator.g_conv_dim=80 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_6.6M &
CUDA_VISIBLE_DEVICES=7 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/results/cifar10/pretrain/BigGAN_10M_trainval model.Generator.g_conv_dim=100 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_10.0M &
CUDA_VISIBLE_DEVICES=3 python run.py public_data.name=null sensitive_data.train_num=val model.ckpt=/results/cifar10/pretrain/BigGAN_14.3M_trainval model.Generator.g_conv_dim=120 -m GS-WGAN -dn cifar10_32 -e 10.0 -ed trainval_14.3M &