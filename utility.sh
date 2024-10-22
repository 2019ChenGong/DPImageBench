# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-Kernel --data_name celeba_male_32 --epsilon 10.0  &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-Kernel --data_name cifar100_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-Kernel --data_name cifar100_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=4  python run.py public_data.name=null --method DP-NTK --data_name cifar100_32 --epsilon 10.0  &

# CUDA_VISIBLE_DEVICES=5  python run.py public_data.name=null --method DP-Kernel --data_name cifar10_32 --epsilon 1.0  &

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

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-ldm/cifar10_32_eps10.0-2024-10-18-07-59-00 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage --data_name cifar100_32 --epsilon 1.0  --exp_path exp/privimage/cifar100_32_eps1.0-2024-10-17-00-27-34 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 10.0  --exp_path exp/privimage/mnist_28_eps10.0-2024-10-19-11-46-32 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/mnist_28_eps1.0-2024-10-19-00-08-31 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name camelyon_32 --epsilon 1.0  --exp_path exp/dp-merf/camelyon_32_eps1.0trainval-2024-10-20-06-35-05 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dp-merf/camelyon_32_eps10.0trainval-2024-10-20-06-32-32 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 1.0  --exp_path exp/dp-merf/celeba_male_32_eps1.0trainval-2024-10-20-06-32-32 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-merf/celeba_male_32_eps10.0trainval-2024-10-20-06-32-32 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 1.0  --exp_path exp/dp-kernel/celeba_male_32_eps1.0trainval-2024-10-20-06-32-32 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-kernel/celeba_male_32_eps10.0trainval-2024-10-20-06-32-32 &

CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name eurosat_32 --epsilon 1.0  --exp_path exp/dp-kernel/eurosat_32_eps1.0trainval-2024-10-20-09-39-26 &

CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dp-kernel/eurosat_32_eps10.0trainval-2024-10-20-09-39-26 &


# CUDA_VISIBLE_DEVICES=4  python eval.py --method PDP-Diffusion --data_name cifar10_32 --epsilon 1.0  --exp_path exp/pdp-diffusion/cifar10_32_eps1.0_LZN-2024-10-20-06-28-03 &

# CUDA_VISIBLE_DEVICES=5  python eval.py --method PDP-Diffusion --data_name cifar10_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/cifar10_32_eps10.0_LZN-2024-10-20-06-25-51 &

# CUDA_VISIBLE_DEVICES=6  python eval.py --method PDP-Diffusion --data_name cifar100_32 --epsilon 1.0  --exp_path exp/pdp-diffusion/cifar100_32_eps1.0_LZN-2024-10-20-07-25-23 &

# CUDA_VISIBLE_DEVICES=7  python eval.py --method PDP-Diffusion --data_name cifar100_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/cifar100_32_eps10.0_LZN-2024-10-20-06-41-36 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage--data_name celeba_male_32 --epsilon 1.0  --exp_path exp/privimage/celeba_male_32_eps1.0-2024-10-17-02-16-40 &
