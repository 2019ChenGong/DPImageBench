CUDA_VISIBLE_DEVICES=4  python run.py public_data.name=null --method DP-NTK --data_name cifar10_32 --epsilon 1.0   &

CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-Kernel --data_name cifar10_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=2  python run.py public_data.name=null --method G-PATE --data_name cifar10_32 --epsilon 1.0   &

CUDA_VISIBLE_DEVICES=3  python run.py public_data.name=null --method DP-MERF --data_name cifar10_32 --epsilon 1.0   &
