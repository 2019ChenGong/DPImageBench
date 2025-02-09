# The results in Figure 5
# method: [DP-MERF, DP-NTK, DP-Kernel, PE, GS-WGAN, DPGAN, DPDM, PDP-Diffusion, DP-LDM-SD, DP-LDM, DP-LORA, PrivImage]
# eps: [0.2, 1.0, 5.0, 10.0, 15.0, 20.0]
data_name=fmnist_28
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


# The results in Figure 6
# method: [DP-MERF, DP-NTK, DP-Kernel, DPGAN, DPDM, PDP-Diffusion, DP-LDM-SD, DP-LDM, DP-LORA, PrivImage]
data_name=cifar10_32
eps=10.0
eval_mode=val

# gan_dim: [40, 60, 80, 100, 120]
gan_dim=40
python run.py setup.n_gpus_per_node=1 public_data.name=null model.Generator.g_conv_dim=$gan_dim --method DP-MERF --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null model.Generator.g_conv_dim=$gan_dim  --method DP-NTK --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 public_data.name=null model.Generator.g_conv_dim=$gan_dim  --method DP-Kernel --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null model.Generator.g_conv_dim=$gan_dim --method DPGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 public_data.name=null model.Generator.g_conv_dim=$gan_dim --method GS-WGAN --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 


# ch_mult: [[2,4],[1,2,3],[1,2,2,4],[1,2,2,4],[1,2,2,4]]
# attn_resolutions: [[16,],[16,8],[16,8,4],[16,8,4],[16,8,4]]
# nf: [32, 64, 64, 96, 128]

ch_mult="[2,4]"
attn_resolutions="[16,]"
nf=32
python run.py setup.n_gpus_per_node=4 public_data.name=null model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method DPDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method PDP-Diffusion --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method DP-LDM-SD --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=4 model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method PrivImage --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 


# ch_mult: [[1,2,2],[1,2,2],[1,2,2,4],[1,2,2,3],[1,2,2,4]]
# attn_resolutions: [[16,8,4],[16,8,4],[16,8,4],[16,8,4],[16,8,4]]
# nf: [32, 64, 64, 96, 128]

ch_mult="[1,2,2]"
attn_resolutions="[16,8,4]"
nf=32
python run.py setup.n_gpus_per_node=1 model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method DP-LDM --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 
python run.py setup.n_gpus_per_node=1 model.network.ch_mult=$ch_mult model.network.attn_resolutions=$attn_resolutions model.network.nf=$nf --method DP-LORA --dataset_name $data_name --epsilon $eps eval.mode=$eval_mode 