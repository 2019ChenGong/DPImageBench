# In RQ3, we investigate the conditional and unconditional pretraining of the studied algorithms using ImageNet as the pretraining dataset.

CUDA_VISIBLE_DEVICES=0,1 python run.py public_data.name=imagenet setup.n_gpus_per_node=2 --epsilon=10.0 pretrain.cond=true --method DP-MERF --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0,1 python run.py public_data.name=imagenet setup.n_gpus_per_node=2 --epsilon=10.0 pretrain.cond=true --method DP-MERF --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0,1 python run.py setup.n_gpus_per_node=2 public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DPGAN --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0,1 python run.py setup.n_gpus_per_node=2 public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DPGAN --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  &

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-Kernel --data_name fmnist_28 eval.mode=val --exp_description val_condition_imagenet  

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=true --method DP-Kernel --data_name cifar10_32 eval.mode=val --exp_description val_condition_imagenet  

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=imagenet --epsilon=10.0 pretrain.cond=false --method DP-LDM --data_name cifar10_32 eval.mode=val --exp_description uncondition_imagenet

# In RQ3, we investigate the pretraining dataset.

CUDA_VISIBLE_DEVICES=0 python run.py public_data.name=places365 --epsilon=10.0 pretrain.cond=true --method DP-MERF --data_name fmnist_28 eval.mode=val public_data.n_classes=365 public_data.train_path=dataset/places365 --exp_description val_condition_places365 