<div align=center>
  
# Question

Perhaps the cluster of Microsoft Research offers a more convenient way to use the ImageNet dataset (it is too large), so Zinan won't need to download it separately. At least Microsoft Research Asia provided this.

# Benchmark on MNIST_28

## 1 Install DPImageBench

 ```
conda create -n dpimagebench python=3.7
conda activate dpimagebench
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorflow-gpu==1.14.0
pip install requirements.txt
git clone git@github.com:2019ChenGong/DPImageBench.git (***We may need to provide the access token***.)
cd DPImageBench
cd opacus; pip install -e .; cd ..
cd models/DPSDA/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..
cd models; gdown https://drive.google.com/uc?id=1yVTWzaSqJVDJy8CsZKtqDoBNeM6154D4; unzip pretrained_models.zip; cd ..
 ```

## 2 Prepare Dataset

 ```
mkdir dataset
cd data; python preprocess_dataset.py --data_name mnist; cd ..
 ```

## 3 Running

 ```
python run.py public_data.name=null --config configs/DP-NTK/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null --config configs/DP-Kernel/mnist_28_eps1.0.yaml
 ```
```
python run.py public_data.name=null --config configs/G-PATE/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null setup.n_gpus_per_node=3 --config configs/DP-LDM/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null --config configs/DP-MERF/mnist_28_eps1.0.yaml
 ```
 ```
python run.py public_data.name=null setup.n_gpus_per_node=3 --config configs/DP-Promise/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null setup.n_gpus_per_node=3 --config configs/DPDM/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null --config configs/DPSDA/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null setup.n_gpus_per_node=3 --config configs/PDP-Diffusion/mnist_28_eps1.0.yaml
 ```
  ```
python run.py public_data.name=null setup.n_gpus_per_node=3 --config configs/PrivImage/mnist_28_eps1.0.yaml
 ```

n_gpus_per_node is the number of GPUs on your node.

All the results will be saved into the folder exp.