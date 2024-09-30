CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-MERF --data_name cifar100_32 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=0  python run.py public_data.name=null --method DP-NTK --data_name fmnist_28 --epsilon 1.0   &

# CUDA_VISIBLE_DEVICES=2  python run.py public_data.name=null --method DP-NTK --data_name fmnist_28 --epsilon 10.0   &

CUDA_VISIBLE_DEVICES=1  python run.py public_data.name=null --method DP-MERF --data_name cifar100_32 --epsilon 10.0   &
