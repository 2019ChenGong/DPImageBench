# pretrain

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval


python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py setup.n_gpus_per_node=3 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval


# finetune


# DPGAN
python run.py public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.cond=true pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.cond=true pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.cond=true pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.cond=true pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-NTK -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval

# DPGAN
python run.py setup.n_gpus_per_node=3 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.cond=true pretrain.n_epochs=400 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.cond=true pretrain.n_epochs=400 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.cond=true pretrain.n_epochs=100 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.cond=true pretrain.n_epochs=100 sensitive_data.train_num=val -m DPGAN -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval

# DP-LDM
python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m DP-LDM -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval

# DP-LDM
python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_goldenIN1_trainval

python run.py setup.n_gpus_per_node=3 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_1.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=400 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_randomIN1_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/golden_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_goldenIN3_trainval

python run.py setup.n_gpus_per_node=2 public_data.selective.ratio=0.01 public_data.selective.semantic_path=/p/fzv6enresearch/DPImageBench/dataset/imagenet/random_semantics_3.pth pretrain.loss.label_unconditioning_prob=0.1 pretrain.n_epochs=100 sensitive_data.train_num=val -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed conditional_randomIN3_trainval

