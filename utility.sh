# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-MERF --data_name camelyon_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-Kernel --data_name celeba_male_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=2  python run.py public_data.name=null --method DP-NTK --data_name camelyon_32 --epsilon 10.0   &

# CUDA_VISIBLE_DEVICES=3 python run.py public_data.name=null --method DP-NTK --data_name camelyon_32 --epsilon 10.0   &

CUDA_VISIBLE_DEVICES=0  python test_classifier.py public_data.name=null --method DP-MERF --data_name cifar10_32 --epsilon 1.0 -ed no-dp-cifar10  &

CUDA_VISIBLE_DEVICES=1  python test_classifier.py public_data.name=null --method DP-MERF --data_name cifar100_32 --epsilon 1.0  -ed no-dp-cifar100 &
