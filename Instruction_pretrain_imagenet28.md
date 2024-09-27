<!-- <div align=center> -->
  
# Question

Perhaps the cluster of Microsoft Research offers a more convenient way to use the ImageNet dataset (it is too large), so Zinan won't need to download it separately. At least Microsoft Research Asia provided this.

# Benchmark on MNIST_28

## 1 Install DPImageBench

 ```
conda create -n dpimagebench_cuda12.1 python=3.9
conda activate dpimagebench_cuda12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
git clone git@github.com:2019ChenGong/DPImageBench.git
cd DPImageBench
pip install -r requirements_cuda12.1.txt
conda install mpi4py
cd opacus; pip install -e .; cd ..
cd models/DPSDA/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..
 ```

## 2 Prepare Dataset

 ```
mkdir dataset; cd dataset
mkdir imagenet; cd imagenet; gdown https://drive.google.com/uc?id=1SFvDfBWmG30xTjFJ0v9Em5avRVL6yqAh; unzip imagenet_32.zip; cd ..; cd ..
 ```

## 3 Running

These methods do not need much GPU Memory and can be run on the same node.

```
python run.py sensitive_data.name=null --method DP-NTK --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null --method DP-Kernel --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null --method DP-MERF --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```

Each of these methods needs nearly 40x3 GB GPU Memory totally.

```
python run.py sensitive_data.name=null setup.n_gpus_per_node=3 --method DP-Promise --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null setup.n_gpus_per_node=3 --method DPDM --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null setup.n_gpus_per_node=3 --method PDP-Diffusion --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null setup.n_gpus_per_node=3 --method PrivImage --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```
```
python run.py sensitive_data.name=null setup.n_gpus_per_node=3 --method DP-LDM --data_name mnist_28 --epsilon 1.0 --exp_description pretrain_imagenet28
```

n_gpus_per_node is the number of GPUs on your node.

All the results will be saved into the folder exp.