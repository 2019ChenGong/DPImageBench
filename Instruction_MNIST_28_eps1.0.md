<!-- <div align=center> -->
  
# Question

Perhaps the cluster of Microsoft Research offers a more convenient way to use the ImageNet dataset (it is too large), so Zinan won't need to download it separately. At least Microsoft Research Asia provided this.

# Benchmark on MNIST_28

## 1 Install DPImageBench

 ```
conda create -n dpimagebench python=3.7
conda activate dpimagebench
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorflow-gpu==1.14.0
git clone git@github.com:2019ChenGong/DPImageBench.git (***We may need to provide the access token***.)
cd DPImageBench
pip install -r requirements.txt
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

These methods do not need much GPU Memory and can be run on the same node.

```
python run.py public_data.name=null --method DP-NTK --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null --method DP-Kernel --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null --method G-PATE --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null --method DP-MERF --data_name mnist_28 --epsilon 1.0
```

Each of these methods needs nearly 40x3 GB GPU Memory totally.

```
python run.py public_data.name=null setup.n_gpus_per_node=3 --method DP-Promise --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null setup.n_gpus_per_node=3 --method DPDM --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null --method DPSDA --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null setup.n_gpus_per_node=3 --method PDP-Diffusion --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null setup.n_gpus_per_node=3 --method PrivImage --data_name mnist_28 --epsilon 1.0
```
```
python run.py public_data.name=null setup.n_gpus_per_node=3 --method DP-LDM --data_name mnist_28 --epsilon 1.0
```

n_gpus_per_node is the number of GPUs on your node.

All the results will be saved into the folder exp.