# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-Kernel --data_name celeba_male_32 --epsilon 10.0  &

# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-NTK --data_name cifar10_32 --epsilon 10.0 sensitive_data.train_num=val -ed trainval &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-NTK --data_name cifar10_32 --epsilon 1.0 sensitive_data.train_num=val -ed trainval &

# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-NTK --data_name cifar10_32 --epsilon 10.0 sensitive_data.train_num=val -ed trainval &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-NTK --data_name cifar10_32 --epsilon 1.0 sensitive_data.train_num=val -ed trainval &

# CUDA_VISIBLE_DEVICES=2  python run.py public_data.name=null --method DP-NTK --data_name cifar100_32 --epsilon 10.0 sensitive_data.train_num=val -ed trainval &

# CUDA_VISIBLE_DEVICES=3  python run.py public_data.name=null --method DP-NTK --data_name cifar100_32 --epsilon 1.0 sensitive_data.train_num=val -ed trainval &


# CUDA_VISIBLE_DEVICES=4  python run.py public_data.name=null --method DP-NTK --data_name cifar100_32 --epsilon 10.0  &

# CUDA_VISIBLE_DEVICES=5  python run.py public_data.name=null --method DP-Kernel --data_name cifar10_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=6  python run.py public_data.name=null --method DP-Kernel --data_name cifar10_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=5  python run.py public_data.name=null --method DP-Kernel --data_name eurosat_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=6  python run.py public_data.name=null --method DP-Kernel --data_name eurosat_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=7  python run.py public_data.name=null --method DP-Kernel --data_name celeba_male_32 --epsilon 1.0   &


# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-NTK --data_name cifar100_32 --epsilon 10.0   &


# CUDA_VISIBLE_DEVICES=3  python run.py public_data.name=null --method DP-NTK --data_name fmnist_28 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-NTK --data_name celeba_male_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-NTK --data_name camelyon_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=2  python run.py public_data.name=null --method DP-NTK --data_name camelyon_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=3  python run.py public_data.name=null --method DP-NTK --data_name fmnist_28 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=7  python run.py public_data.name=null --method DP-MERF --data_name cifar10_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=0  python test_classifier.py public_data.name=null --method DP-MERF --data_name mnist_28 --epsilon 1.0  -ed a-no-dp-eurosat_32 &

# CUDA_VISIBLE_DEVICES=1  python test_classifier.py public_data.name=null --method DP-MERF --data_name fmnist_28 --epsilon 1.0 -ed a-no-dp-fmnist  &

# CUDA_VISIBLE_DEVICES=0  python test_classifier.py public_data.name=null --method DP-MERF --data_name cifar10_32 --epsilon 1.0  -ed a-no-dp-cifar10_32-edit &

# CUDA_VISIBLE_DEVICES=1  python test_classifier.py public_data.name=null --method DP-MERF --data_name cifar100_32 --epsilon 1.0 -ed a-no-dp-cifar100_32-edit  &

# CUDA_VISIBLE_DEVICES=0  python test_classifier.py public_data.name=null --method DP-MERF --data_name camelyon_32 --epsilon 1.0  -ed a-no-dp-camelyon_32 &

# CUDA_VISIBLE_DEVICES=1  python test_classifier.py public_data.name=null --method DP-MERF --data_name celeba_male_32 --epsilon 1.0 -ed a-no-dp-celeba_male_32  &

# CUDA_VISIBLE_DEVICES=2  python test_classifier.py public_data.name=null --method DP-MERF --data_name eurosat_32 --epsilon 1.0 -ed a-no-dp-eurosat_32  &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-NTK --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-ntk/cifar10_32_eps10.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-NTK --data_name cifar10_32 --epsilon 1.0  --exp_path exp/dp-ntk/cifar10_32_eps1.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-NTK --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-ntk/cifar100_32_eps10.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-NTK --data_name cifar100_32 --epsilon 1.0  --exp_path exp/dp-ntk/cifar100_32_eps1.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name cifar10_32 --epsilon 1.0  --exp_path exp/dpdm/cifar10_32_eps1.0trainval-2024-10-23-14-01-14 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DPDM sensitive_data.train_num=val --data_name cifar10_32 --epsilon 1.0  --exp_path exp/dpdm/cifar10_32_eps10.0trainval-2024-10-24-01-44-41 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM --data_name celeba_male_32 --epsilon 1.0 --exp_path exp/dp-ldm/celeba_male_32_eps1.0unconditional-2024-10-25-20-13-02 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-LDM --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-ldm/celeba_male_32_eps10.0unconditional-2024-10-25-20-11-00 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-LDM --data_name cifar100_32 --epsilon 10.0  sensitive_data.train_num=val  --exp_path exp/dp-ldm/cifar100_32_eps10.0unconditional_trainval-2024-10-26-00-06-28 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-LDM --data_name cifar100_32 --epsilon 1.0 sensitive_data.train_num=val  --exp_path exp/dp-ldm/camelyon_32_eps10.0unconditional_trainval-2024-10-25-23-46-23 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-LDM --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-ldm/cifar100_32_eps10.0unconditional-2024-10-25-07-00-22 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dp-ldm/mnist_28_eps1.0unconditional-2024-10-24-22-05-50 &

# CUDA_VISIBLE_DEVICES=7  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/mnist_28_eps1.0unconditional-2024-10-24-22-05-50 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM --data_name cifar100_32 --epsilon 1.0  --exp_path exp/dp-ldm/cifar100_32_eps1.0unconditional-2024-10-26-13-44-11 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-LDM --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dp-ldm/fmnist_28_eps10.0unconditional-2024-10-24-20-59-35 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/mnist_28_eps10.0-2024-10-24-09-51-58 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage sensitive_data.train_num=val --data_name mnist_28 --epsilon 1.0  --exp_path exp/privimage/mnist_28_eps1.0trainval-2024-10-27-20-44-57 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method PrivImage sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/privimage/celeba_male_32_eps10.0trainval-2024-10-26-10-44-41 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/eurosat_32_eps10.0_trainval_LZN-2024-10-28-02-54-55 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/eurosat_32_eps10.0_trainval_LZN-2024-10-28-02-54-55 &


# CUDA_VISIBLE_DEVICES=0  python eval.py --method PDP-Diffusion --data_name cifar10_32 --epsilon 1.0 sensitive_data.train_num=val  --exp_path exp/pdp-diffusion/cifar10_32_eps1.0_trainval_LZN-2024-10-27-02-23-40 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method PDP-Diffusion --data_name cifar10_32 --epsilon 10.0 sensitive_data.train_num=val --exp_path exp/pdp-diffusion/cifar10_32_eps10.0_trainval_LZN-2024-10-27-02-23-55 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method PDP-Diffusion --data_name cifar100_32 --epsilon 1.0 sensitive_data.train_num=val  --exp_path exp/pdp-diffusion/cifar100_32_eps1.0_trainval_LZN-2024-10-27-02-24-44 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method PDP-Diffusion --data_name cifar100_32 --epsilon 10.0 sensitive_data.train_num=val  --exp_path exp/pdp-diffusion/cifar100_32_eps10.0_trainval_LZN-2024-10-27-02-24-11 &


# CUDA_VISIBLE_DEVICES=3  python eval.py --method DPDM sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dpdm/eurosat_32_eps10.0trainval-2024-10-24-12-56-31 &

# CUDA_VISIBLE_DEVICES=4  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 1.0  --exp_path exp/dpgan/cifar10_32_eps1.0trainval-2024-10-22-21-23-15 &

# CUDA_VISIBLE_DEVICES=5  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dpgan/cifar10_32_eps10.0trainval-2024-10-22-21-58-29 &

# CUDA_VISIBLE_DEVICES=6  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar100_32 --epsilon 1.0  --exp_path exp/dpgan/cifar100_32_eps1.0trainval-2024-10-22-21-37-33 &

# CUDA_VISIBLE_DEVICES=7  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dpgan/cifar100_32_eps10.0trainval-2024-10-22-21-35-25 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage--data_name celeba_male_32 --epsilon 1.0  --exp_path exp/privimage/celeba_male_32_eps1.0-2024-10-17-02-16-40 &

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py setup.n_gpus_per_node=4 --method DPGAN public_data.name=null --epsilon 10.0 -dn cifar10_32 sensitive_data.train_num=val  -ed trainval

# python run.py setup.n_gpus_per_node=4 --method DP-LDM --epsilon 1.0 -dn mnist_28 sensitive_data.train_num=val  -ed unconditional

# python run.py setup.n_gpus_per_node=3 --method DP-LDM --epsilon 1.0 -dn cifar100_32 -ed unconditional pretrain.loss.label_unconditioning_prob=1.0

# CUDA_VISIBLE_DEVICES=7  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name mnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/mnist_28_eps1.0unconditionalval-2024-10-25-14-41-07 &


CUDA_VISIBLE_DEVICES=0  python eval.py --method PE --data_name mnist_28 --epsilon 1.0  --exp_path exp/pe/mnist_28_eps1.0_th24_tr10_vds000011112222-2024-10-26-18-18-02 &

CUDA_VISIBLE_DEVICES=1  python eval.py --method PE --data_name fmnist_28 --epsilon 1.0  --exp_path exp/pe/fmnist_28_eps1.0_th24_tr10_vds000011112222-2024-10-28-17-12-41 &

CUDA_VISIBLE_DEVICES=2  python eval.py --method PE sensitive_data.train_num=val --data_name mnist_28 --epsilon 1.0  --exp_path exp/pe/mnist_28_eps1.0_trainval_th24_tr10_vds000011112222-2024-10-29-09-40-33 &

CUDA_VISIBLE_DEVICES=3  python eval.py --method PE sensitive_data.train_num=val --data_name fmnist_28 --epsilon 1.0  --exp_path exp/pe/fmnist_28_eps1.0_trainval_th24_tr10_vds000011112222-2024-10-29-10-44-36 &
