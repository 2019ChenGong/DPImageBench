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

# CUDA_VISIBLE_DEVICES=1  python eval.py --method PrivImage --data_name cifar100_32 --epsilon 10.0  --exp_path exp/privimage/cifar100_32_eps10.0-2024-10-17-00-29-00 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/mnist_28_eps1.0-2024-10-19-00-08-31 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPDM --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dpdm/camelyon_32_eps10.0-2024-10-10-20-39-43 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-LDM --data_name mnist_28 --epsilon 10.0  --exp_path exp/dp-ldm/mnist_28_eps10.0-2024-10-19-11-46-32 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method PrivImage --data_name cifar10_32 --epsilon 10.0  --exp_path exp/privimage/cifar10_32_eps10.0-2024-10-16-17-28-44 &

# CUDA_VISIBLE_DEVICES=1  python eval.py public_data.name=null --method DPGAN --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dpgan/celeba_male_32_eps10.0-2024-10-08-12-09-30 &

CUDA_VISIBLE_DEVICES=6  python eval.py --method DP-LDM --data_name fmnist_28 --epsilon 1.0  --exp_path exp/dp-ldm/fmnist_28_eps1.0-2024-10-19-16-56-37 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPSDA --data_name mnist_28 --epsilon 1.0  --exp_path exp/dpsda/mnist_28_eps10.0_th0_tr10_vds000111222333 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DPSDA --data_name mnist_28 --epsilon 1.0  --exp_path exp/dpsda/mnist_28_eps10.0_th2_tr10_vds00112233 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DPSDA --data_name mnist_28 --epsilon 1.0  --exp_path exp/dpsda/mnist_28_eps10.0_th2.5_tr10_vds000111222333 &