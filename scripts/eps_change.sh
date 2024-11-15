# sh scripts/eps_change.sh

# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 1.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-MERF -dn cifar10_32 -e 1.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-MERF -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-MERF -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-MERF -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-MERF -dn cifar10_32 -e 10.0 &


# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.n_splits=1 train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 1.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 1.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-NTK -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 &

# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 1.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 1.0 &
# CUDA_VISIBLE_DEVICES=0 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=1 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 10.0 &
# CUDA_VISIBLE_DEVICES=2 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-Kernel -dn fmnist_28 -e 10.0 &
# CUDA_VISIBLE_DEVICES=2 python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DP-Kernel -dn cifar10_32 -e 10.0 &

# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 1.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 1.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DPGAN -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0

# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DPDM -dn fmnist_28 -e 1.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 1.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DPDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DPDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DPDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 public_data.name=null train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m DPDM -dn cifar10_32 -e 10.0

# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps0.2 sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps0.2 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps5 sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps5 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps15 sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps15 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps20 sensitive_data.train_num=val -m PDP-Diffusion -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps20 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0

# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps0.2 sensitive_data.train_num=val -m DP-LDM -dn fmnist_28 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps0.2 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps5 sensitive_data.train_num=val -m DP-LDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps5 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps15 sensitive_data.train_num=val -m DP-LDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps15 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps20 sensitive_data.train_num=val -m DP-LDM -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 pretrain.loss.label_unconditioning_prob=1.0 -ed unconditional_trainval_eps20 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0

# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=0.2 -ed trainval_eps0.2 sensitive_data.train_num=val -m PrivImage -dn cifar10_32 -e 1.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=5 -ed trainval_eps5 sensitive_data.train_num=val -m PrivImage -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=15 -ed trainval_eps15 sensitive_data.train_num=val -m PrivImage -dn cifar10_32 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m PrivImage -dn fmnist_28 -e 10.0
# python run.py setup.n_gpus_per_node=3 train.dp.epsilon=20 -ed trainval_eps20 sensitive_data.train_num=val -m PrivImage -dn cifar10_32 -e 10.0