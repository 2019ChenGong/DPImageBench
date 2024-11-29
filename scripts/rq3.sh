# In RQ3, we investigate the conditional and unconditional pretraining of the studied algorithms using ImageNet as the pretraining dataset.

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-MERF --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-MERF --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0,1 python run.py setup.n_gpus_per_node=2 public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DPGAN --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0,1 python run.py setup.n_gpus_per_node=2 public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DPGAN --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-Kernel --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  

CUDA_VISIBLE_DEVICES=1 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-Kernel --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  
