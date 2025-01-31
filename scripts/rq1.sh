# The results in Table 6 and 7
# method: [DP-MERF, DP-NTK, DP-Kernel, PE, GS-WGAN, DPGAN, DPDM, PDP-Diffusion, DP-LDM-SD, DP-LDM, DP-LORA, PrivImage]
# data_name: [mnist_28, fmnist_28, cifar10_32, cifar100_32, eurosat_32, celeba_male_32, camelyon_32]
# eps: [10.0, 1.0]
# eval_mode: [val, ses]
data_name=mnist_28
eps=10.0
eval_mode=val
python run.py setup.n_gpus_per_node=1 public_data.name=null --method DP-MERF --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null  --method DP-NTK --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null  --method DP-Kernel --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PE --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method GS-WGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method DPGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method DPDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method DP-LDM-SD --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 --method DP-LDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 --method DP-LORA --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PrivImage --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 


# The results in Table 9
# resolution: [32, 64, 128]
resolution=32
data_name='celeba_male_'$resolution
eps=10.0
eval_mode=val
python run.py setup.n_gpus_per_node=1 public_data.name=null --method DP-MERF --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null  --method DP-NTK --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null  --method DP-Kernel --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PE --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method GS-WGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method DPGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null --method DPDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method DP-LDM-SD --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 --method DP-LDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 --method DP-LORA --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 --method PrivImage --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 