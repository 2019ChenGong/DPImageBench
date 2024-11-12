# CUDA_VISIBLE_DEVICES=5  python eval.py --method DPDM sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dpdm/mnist_28_eps10.0trainval-2024-10-22-22-42-07 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dpdm/fmnist_28_eps10.0trainval-2024-10-23-19-31-12 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DPDM sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dpdm/cifar10_32_eps10.0trainval-2024-10-24-01-44-41 -op fidelity&

# CUDA_VISIBLE_DEVICES=6  python eval.py --method DPDM sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dpdm/cifar100_32_eps10.0trainval-2024-10-26-01-37-36 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPDM sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dpdm/eurosat_32_eps10.0trainval-2024-10-24-12-56-31 &

# CUDA_VISIBLE_DEVICES=7  python eval.py --method DPDM sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dpdm/celeba_male_32_eps10.0trainval-2024-10-24-00-28-59 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DPDM sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dpdm/camelyon_32_eps10.0trainval-2024-10-25-02-30-03 -op fidelity&

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DPGAN sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dpgan/mnist_28_eps10.0trainval-2024-10-22-11-42-58 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dpgan/fmnist_28_eps10.0trainval-2024-10-22-11-44-30 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dpgan/cifar10_32_eps10.0trainval-2024-10-22-21-58-29 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dpgan/cifar100_32_eps10.0trainval-2024-10-22-21-35-25 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DPGAN sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dpgan/eurosat_32_eps10.0trainval-2024-10-22-12-46-17 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPGAN sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dpgan/celeba_male_32_eps10.0trainval-2024-10-22-14-42-52 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DPGAN sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dpgan/camelyon_32_eps10.0trainval-2024-10-22-22-13-03 &

# CUDA_VISIBLE_DEVICES=5  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/pdp-diffusion/mnist_28_eps10.0_trainval_LZN-2024-10-26-19-37-06 &

# CUDA_VISIBLE_DEVICES=5  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/pdp-diffusion/fmnist_28_eps10.0_trainval_LZN-2024-10-26-21-07-52 &

# CUDA_VISIBLE_DEVICES=6  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/cifar10_32_eps10.0_trainval_LZN-2024-10-27-02-23-55 &

# CUDA_VISIBLE_DEVICES=6  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/cifar100_32_eps10.0_trainval_LZN-2024-10-27-02-24-11 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/eurosat_32_eps10.0_trainval_LZN-2024-10-28-02-54-55 &

# CUDA_VISIBLE_DEVICES=6  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/celeba_male_32_eps10.0trainval-2024-10-22-14-42-52 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method PDP-Diffusion sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/pdp-diffusion/camelyon_32_eps10.0trainval-2024-10-22-22-13-03 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dp-merf/mnist_28_eps10.0trainval-2024-10-19-06-42-21 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dp-merf/fmnist_28_eps10.0trainval-2024-10-20-06-27-04 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-merf/cifar10_32_eps10.0trainval-2024-10-20-06-27-04 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-merf/cifar100_32_eps10.0trainval-2024-10-20-06-30-09 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dp-merf/eurosat_32_eps10.0trainval-2024-10-20-06-30-09 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-merf/celeba_male_32_eps10.0trainval-2024-10-22-03-29-00 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-MERF sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dp-merf/camelyon_32_eps10.0trainval-2024-10-20-06-32-32 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dp-ntk/mnist_28_eps10.0trainval-2024-10-20-09-43-52 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dp-ntk/fmnist_28_eps10.0trainval-2024-10-20-09-45-58 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-ntk/cifar10_32_eps10.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-ntk/cifar100_32_eps10.0trainval-2024-10-24-12-30-04 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dp-ntk/eurosat_32_eps10.0trainval-2024-10-24-01-48-00 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-ntk/celeba_male_32_eps10.0trainval-2024-10-23-07-28-01 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-NTK sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dp-ntk/camelyon_32_eps10.0trainval-2024-10-23-07-26-25 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dp-kernel/mnist_28_eps10.0trainval-2024-10-20-06-44-57 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dp-kernel/fmnist_28_eps10.0trainval-2024-10-20-06-44-57 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-kernel/cifar10_32_eps10.0trainval-2024-10-20-09-38-09 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-kernel/cifar100_32_eps10.0trainval-2024-10-22-14-48-27 &

# CUDA_VISIBLE_DEVICES=4  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/dp-kernel/eurosat_32_eps10.0trainval-2024-10-20-09-39-26 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/dp-kernel/celeba_male_32_eps10.0trainval-2024-10-22-08-00-43 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/dp-kernel/camelyon_32_eps10.0trainval-2024-10-20-09-41-01 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/privimage/mnist_28_eps10.0trainval-2024-10-27-10-13-22 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method PrivImage sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/privimage/fmnist_28_eps10.0trainval-2024-10-27-12-12-01 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method PrivImage sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/privimage/cifar10_32_eps10.0trainval-2024-10-27-08-10-58 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method PrivImage sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/privimage/cifar100_32_eps10.0trainval-2024-10-27-01-33-21 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method PrivImage sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/privimage/celeba_male_32_eps10.0trainval-2024-10-26-10-44-41 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method PrivImage sensitive_data.train_num=val --data_name eurosat_32 --epsilon 10.0  --exp_path exp/privimage/eurosat_32_eps10.0trainval-2024-10-28-15-06-50 &

# CUDA_VISIBLE_DEVICES=3  python eval.py --method PrivImage sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/privimage/camelyon_32_eps10.0trainval-2024-10-26-10-32-04 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name mnist_28 --epsilon 10.0  --exp_path exp/dp-ldm/mnist_28_eps10.0unconditionalval-2024-10-25-16-48-30 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path exp/dp-ldm/fmnist_28_eps10.0unconditionalval-2024-10-25-16-51-25 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path exp/dp-ldm/cifar10_32_eps10.0unconditionalval-2024-10-26-02-03-24 &

# CUDA_VISIBLE_DEVICES=1  python eval.py --method DP-LDM sensitive_data.train_num=val --data_name cifar100_32 --epsilon 10.0  --exp_path exp/dp-ldm/cifar100_32_eps10.0unconditional_trainval-2024-10-26-00-06-28 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name celeba_male_32 --epsilon 10.0  --exp_path exp/gs-wgan/celeba_male_32_eps10.0trainval_step5w-2024-11-04-21-58-52 &

# CUDA_VISIBLE_DEVICES=2  python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name camelyon_32 --epsilon 10.0  --exp_path exp/gs-wgan/camelyon_32_eps10.0trainval-2024-11-04-21-49-24 &

# CUDA_VISIBLE_DEVICES=0  python eval.py --method GS-WGAN --data_name mnist_28 --epsilon 10.0  --exp_path exp/gs-wgan/mnist_28_eps10.0-2024-10-28-04-43-28 &

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps1.0trainval_eps0.2-2024-10-29-05-30-19

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps1.0trainval-2024-10-20-06-44-57

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval_eps5-2024-10-29-05-30-19

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval_eps15-2024-10-29-05-30-19

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-Kernel sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-kernel/fmnist_28_eps10.0trainval_eps20-2024-10-29-05-30-19

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ldm/fmnist_28_eps1.0unconditional_trainval_eps0.2-2024-10-29-07-32-19

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ldm/fmnist_28_eps1.0unconditional_trainval-2024-10-25-21-06-37

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ldm/fmnist_28_eps10.0unconditional_trainval_eps5-2024-10-30-06-49-45

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ldm/fmnist_28_eps10.0unconditional_trainval_eps15-2024-10-29-07-32-49

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-LDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ldm/fmnist_28_eps10.0unconditional_trainval_eps20-2024-10-29-15-06-15

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps1.0trainval_eps0.2-2024-10-29-00-07-10

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps1.0trainval-2024-10-20-06-27-04

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_eps5-2024-10-29-00-07-10

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_eps15-2024-10-29-00-07-10

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_eps20-2024-10-29-00-07-10

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-MERF sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-merf/fmnist_28_eps10.0trainval_eps20-2024-10-29-00-07-10

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ntk/fmnist_28_eps1.0trainval_eps0.2-2024-10-29-04-04-11

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ntk/fmnist_28_eps1.0trainval-2024-10-20-09-45-58

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ntk/fmnist_28_eps10.0trainval_eps5-2024-10-29-04-04-11

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ntk/fmnist_28_eps10.0trainval_eps15-2024-10-29-04-03-59

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DP-NTK sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dp-ntk/fmnist_28_eps10.0trainval_eps20-2024-10-29-04-03-59

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpdm/fmnist_28_eps1.0trainval_eps0.2-2024-10-29-01-51-05

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpdm/cifar10_32_eps1.0trainval-2024-10-23-14-01-14

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpdm/cifar10_32_eps10.0trainval_eps5-2024-10-28-04-31-08

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpdm/cifar10_32_eps10.0trainval_eps15-2024-10-28-18-26-48

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPDM sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpdm/cifar10_32_eps10.0trainval_eps20-2024-10-29-01-09-20

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps1.0trainval_eps0.2-2024-10-28-23-46-11

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps1.0trainval-2024-10-22-15-52-27

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps5-2024-10-29-04-05-26

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps15-2024-10-29-04-30-41

# CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0trainval_eps20-2024-10-29-04-35-29

CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval_6.6M-2024-11-06-00-53-47

CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval_10M-2024-11-06-01-43-51

CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/cifar10_32_eps10.0trainval_14.3M-2024-11-07-05-24-54

CUDA_VISIBLE_DEVICES=0 python eval.py --method DPGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/dpgan/fmnist_28_eps10.0unconditional_imagenet_trainval-2024-11-09-07-55-28



CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/cifar10_32_eps10.0trainval_3.8M-2024-11-07-23-22-03

CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/cifar10_32_eps10.0trainval_6.6M-2024-11-07-23-22-03

CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name cifar10_32 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/cifar10_32_eps10.0trainval_10.0M-2024-11-07-23-21-16

CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/fmnist_28_eps10.0conditional_imagenet_trainval_1.4M-2024-11-10-01-51-26

CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/fmnist_28_eps10.0trainval_1.4M-2024-11-10-01-48-10

CUDA_VISIBLE_DEVICES=0 python eval.py --method GS-WGAN sensitive_data.train_num=val --data_name fmnist_28 --epsilon 10.0  --exp_path /p/fzv6enresearch/DPImageBench/exp/gs-wgan/fmnist_28_eps10.0unconditional_imagenet_trainval_1.4M-2024-11-10-01-48-59

