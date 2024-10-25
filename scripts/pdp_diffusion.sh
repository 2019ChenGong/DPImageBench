python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn mnist_28 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn fmnist_28 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar10_32 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar100_32 -e 10.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar100_32 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn eurosat_32 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn celeba_male_32 -e 10.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn celeba_male_32 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn camelyon_32 -e 10.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn camelyon_32 -e 1.0 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0

python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn mnist_28 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn mnist_28 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn fmnist_28 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn fmnist_28 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar10_32 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar10_32 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar100_32 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn cifar100_32 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn eurosat_32 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn eurosat_32 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn celeba_male_32 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn celeba_male_32 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn camelyon_32 -e 10.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 
python run.py setup.n_gpus_per_node=4 -m PDP-Diffusion -dn camelyon_32 -e 1.0 -ed unconditional_trainval pretrain.loss.label_unconditioning_prob=1.0 sensitive_data.train_num=val 