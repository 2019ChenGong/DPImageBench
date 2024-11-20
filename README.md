<div align=center>
  
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis Algorithms
</div>

DPImageBench is an open-source toolkit developed to facilitate the research and application of DP image synthesis. DPImageBench simplifies the access, understanding, and assessment of DP image synthesis, making it accessible to both researchers and the broader community.

<div align=center>
<img src="./plot/figures/eps10_visual.png" width = "1000" alt="Synthetic images by algorithms in DPImageBench with epsilon=10" align=center />
</div>

<p align="center">Synthetic images by algorithms in DPImageBench with $\epsilon=10$ .</p>

## 1. Contents
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
    - [2.1 Currently Supported Algorithms](#21-currently-supported-algorithms)
    - [2.2 Currently Supported Datasets](#22-currently-supported-datasets)
  - [3. Repo Contents](#3-repo-contents)
  - [4. Quick Start](#4-quick-start)
    - [4.1 Install DPImageBench](#41-install-dpimagebench)
    - [4.2 Prepare Dataset](#42-prepare-dataset)
    - [4.3 Running](#43-running)
    - [4.4 Results Explanation](#44-results-explanation)
  - [5. Customization](#5-customization)
  - [6. Contacts](#6-contacts)
  - [Acknowledgment](#acknowledgement)

### Updates 
- 🎉 **(2024.11.19)** We're thrilled to announce the release of initial version of DPImageBench!

### Todo

- [ ] setup.master_port=6026

- [ ] remove the unneccessary part for algorithms in models

- [ ] n_split, unify the batchsize?

- [-] I found that changing the generator of GS-WGAN into ours affects the performance a lot. [Testing]

- [ ] The model size in Biggan is different from DMs. 

- [ ] Optimizing utils, torch_utils, dnnlib. 


## 2. Introduction

### 2.1 Currently Supported Algorithms

We list currently supported DP image synthesis methods as follows.

  | Methods |  Link                                                         |
  | -------------- | ------------------------------------------------------------ |
  | DP-MERF            |  [\[AISTATS 2021\] DP-MERF: Differentially Private Mean Embeddings With Randomfeatures for Practical Privacy-Preserving Data Generation](https://proceedings.mlr.press/v130/harder21a.html) |
  | DP-NTK            |  [\[AISTATS 2021\] Differentially Private Neural Tangent Kernels (DP-NTK) for Privacy-Preserving Data Generation](https://arxiv.org/html/2303.01687v2) |
  | DP-Kernel        |  [\[NeuriPS 2023\] Functional Renyi Differential Privacy for Generative Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html) |
  | PE          |  [\[ICLR 2024\] Differentially Private Synthetic Data via Foundation Model {API}s 1: Images](https://openreview.net/forum?id=YEhQs8POIo) |
  | GS-WGAN            |  [\[NeuriPS 2020\] GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators](https://arxiv.org/pdf/2006.08265) |
  | DP-GAN            |  [\[1802.06739\] Differentially Private Generative Adversarial Network (arxiv.org)](https://arxiv.org/abs/1802.06739) |
  | DPDM          |  [\[TMLR 2023\] Differentially Private Diffusion Models](https://openreview.net/forum?id=ZPpQk7FJXF) |
  | PDP-Diffusion       | [\[2302.13861\] Differentially Private Diffusion Models Generate Useful Synthetic Images (arxiv.org)](https://arxiv.org/abs/2302.13861) |
  | DP-LDM            | [\[TMLR 2024\] Differentially Private Latent Diffusion Models (arxiv.org)](https://arxiv.org/abs/2305.15759)            |
  | PrivImage       | [\[UESNIX Security 2024\] PrivImage: Differentially Private Synthetic Image Generation using Diffusion Models with Semantic-Aware Pretraining](https://www.usenix.org/conference/usenixsecurity24/presentation/li-kecen) |

### 2.2 Currently Supported Datasets
We list the studied datasets as follows, which include seven sensitive datasets and two public datasets.
  | Usage |  Dataset  |
  | ------- | --------------------- |
  | Pretraining dataset | ImageNet_ILSVRC2012, Places365 |
  | Sensitive dataset | MNIST, FashionMNIST, CIFAR-10, CIFAR-100, EuroSAT, CelebA, Camelyon |

## 3. Repo Contents

Below is the directory structure of the DPImageBench project, which organizes its two core functionalities within the `models/` and `evaluation/` directories. To enhance user understanding and showcase the toolkit's ease of use, we offer a variety of example scripts located in the `scripts/` directory.


```plaintext
DPImageBench/
├── config/                     # Configuration files for various DP image synthesis algorithms
│   ├── DP-MERF      
│   ├── DP-NTK       
│   ├── DP-Kernel
│   ├── PE            
│   ├── DP-GAN         
│   ├── DPDM        
│   ├── PDP-Diffusion      
│   ├── DP-LDM   
│   ├── GS-WGAN
│   └── PDP-Diffusion   
├── data/                       # Data Preparation for Our Benchmark
│   ├── stylegan3
│   ├── SpecificPlaces365.py
│   ├── dataset_loader.py
│   └── preprocess_dataset.py 
├── dataset/                    # Datasets studied in the project
├── dnnlib/ 
├── docker/                     # Docker file
├── exp/                        # The output of the training process and evaluation results.
├── evaluation/                 # Evaluation module of DPImageBench, including utility and fidelity
│   ├── classifier/             # Downstream tasks classification training algorithms
│   │   ├── densenet.py  
│   │   ├── resnet.py 
│   │   ├── resnext.py 
│   │   └── wrn.py 
│   ├── ema.py 
│   └── evaluator.py 
├── models/                     # Implementation framework for DP image synthesis algorithms
│   ├── DP_Diffusion      
│   ├── DP_GAN       
│   ├── DP_MERF
│   ├── DP_NTK          
│   ├── GS_WGAN       
│   ├── PE     
│   ├── PrivImage
│   ├── dpsgd_diffusion.py
│   ├── dpsgd_gan.py
│   ├── pretrained_models       # The pre-downloaed files for PE and PrivImage
│   ├── model_loader.py           
│   └── synthesizer.py  
├── opacus/                     # Implementation of DPSGD
├── plot/                       # Figures and plots in our paper
│   ├── plot_eps_change.py                           # Plotting for Figure 5 and 10
│   ├── plot_size_change.py                          # Plotting for Figure 6
│   ├── plot_wo_pretrain_cond_cifar10.py             # Plotting for Figure 7
│   ├── plot_wo_pretrain_cond_fmnist.py              # Plotting for Figure 9
│   ├── plot_wo_pretrain_places_imagenet.py          # Plotting for Figure 4   
│   └── visualization.py   
├── scripts/                    # Scripts for using DPImageBench
│   ├── diffusion_size_change.py                    
│   ├── download_dataset.sh                          
│   ├── eps_change.sh.                               
│   ├── gan_size_change.sh                           
│   ├── pdp_diffusion.sh                             
│   └── test_classifier.py                           
├── torch_utils/                # Helper classes and functions supporting various operations
│   └── persistence.py                     
├── utils/                      # Helper classes and functions supporting various operations
│   └── utils.py                     
├── README.md                   # Main project documentation
└── requirements.txt            # Dependencies required for the project
```

## 4. Quick Start

### 4.1 Install DPImageBench

Clone repo and setup the environment:

 ```
git clone git@github.com:2019ChenGong/DPImageBench.git
sh install.sh
 ```

We also provide the (Docker)[./docker/Dockerfile] file.

### 4.2 Prepare Dataset

 ```
sh scripts/data_preparation.sh
 ```

After running, we can found the folder `dataset`:

  ```plaintext
dataset/                                  
├── camelyon/       
├── celeba/ 
├── cifar10/ 
...
```

### 4.3 Running

The training and evaluatin codes are `run.py` and `eval.py`.

#### 4.3.1 Key hyper-parameter introductions.

We list the key hyper-parameters below, including their explanations and available options.

- `--dataset_name`: means the sensitive dataset; the option is [`mnist_28`, `fmnist_28`, `cifar10_32`, `cifar100_32`, `eurosat_32`, `celeba_male_32`, `camelyon_32`].
- `--method`: the method to train the DP image synthesizers; the option is [`DP-NTK`, `DP-Kernel`, `DP-MERF`, `DPGAN`, `DP-LDM`, `DPDM`, `PE`, `GS-WGAN`, `PDP-Diffusion`, `PrivImage`].
- `--epsilon`: the privacy budget 10.0; the option is [`1.0`, `10.0`].
- `--exp_description`: the notes for the name of result folders.
- `setup.n_gpus_per_node`: means the number of GPUs to be used for training.
- `pretrain.cond`: specifies the mode of pretraining. The options are [`true`, `false`], where `true` indicates conditional pretraining and `false` indicates conditional pretraining.
- `public_data.name`: the name of pretraining dataset; the option is [`null`, `imagenet`, `places365`], which mean that without pretraining, using ImageNet dataset as pretraining dataset, and using Places365 as pretraining dataset. It is notice that DPImageBench uses ImageNet as default pretraining dataset. If users use Places365 as pretraining dataset, please add `public_data.n_classes=365 public_data.train_path=dataset/places365`.
- `eval.mode`: the mode of evaluations; the option is [`val`, `syn`] which means that using part of sensitive images and directly using the synthetic images as the validation set for model selection, respectively. The default setting is `val`.


#### 4.3.2 How to run.

Users should first activate the conda environment.

```
conda activate dpimagebench
cd DPImageBench
```
#### For the implementation of results reported in Table 5, 6, and 7 (RQ1). 

We list an example as follows. Users can modify the configuration files in [configs](./configs) as their preference. 

We provide an example of training a synthesizer using the PDP-Diffusion method with 4 GPUs. The results reported in Table 6 were obtained by following the instructions provided. Additionally, the results (fidelity evaluations) reported in Table 7 were obtained using the default settings.
```
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name mnist_28 --epsilon 10.0 eval.mode=val
```
The results reported in Table 5 were obtained by following the instructions below.
```
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name mnist_28 --epsilon 10.0 eval.mode=syn
```
We provide more examples in the `scripts/rq1.sh`, please refer to [scrips](scripts/rq1.sh).

Besides, if users want to directly evaluate the synthetic images,
```
python eval.py --method PDP-Diffusion --data_name mnist_28 --epsilon 10.0 --exp_path exp/pdp-diffusion/<the-name-of-file>
```
The results are recorded in `exp/pdp-diffusion/<the-name-of-file>/stdout.txt`.

Test the classification algorithm on the sensitive images without DP.
```
python ./scripts/test_classifier.py --method PDP-Diffusion --data_name mnist_28 --epsilon 10.0  -ed no-dp-mnist_28
```
The results are recorded in `exp/pdp-diffusion/<the-name-of-file>no-dp-mnist_28/stdout.txt`. This process is independent of `--method` and uses of `--epsilon`.

#### Directly use the pretrained synthesizers.

If users wish to finetune the synthesizers using pretrained models, they should: (1) set `public_data.name=null`, and (2) load the pretrained synthesizers through `model.ckpt`. For example, the pretrained synthesizer can be sourced from other algorithms. Readers can refer to the [file structure](./exp/README.md) for more details about loading pretrained models.

```
python run.py setup.n_gpus_per_node=3 public_data.name=null eval.mode=val \
 model.ckpt=./exp/pdp-diffusion/<the-name-of-scripts>/pretrain/checkpoints/final_checkpoint.pth \
 --method PDP-Diffusion --dataset_name fmnist_28 --epsilon 10.0 --exp_description <any-notes>
```


#### For the implementation of the results reported in Figures 5, 6, and 9 (RQ2), the performance is analyzed by varying the epsilon and model size.

If users wish to change the size of the synthesizer, the following parameters should be considered.

- `train.dp.n_split`: the number of gradient accumulations. For example, if you set `batch_size` as 500, but your server only allows the max `batch_size` 250, you can set `train.dp.n_split` as 2.
- Change the model size: For diffusion based model, `model.network.ch_mult` is a list of positive integers, which determines the model size. By default, `model.network.ch_mult` is [2,2]. You can increase the model size through increasing its depth and width. To increase the depth, you can extend this list by `model.network.ch_mult=[2,2,2]`. To increase the width, you can increase the integers in the list by `model.network.ch_mult=[4,4]`. For GAN based model, please change `model.Generator.g_conv_dim=100` to adjust the synthesizer size.

For example:


#### For the implementation of the results reported in RQ3.

Users can set the `pretrain.cond` and `public_data.name` to choose between conditional and unconditional pretraining or to enable or disable pretraining. `public_data.name=null` indicates that pretraining is excluded. If users wish to use Places365 or a pretraining dataset, please take note of the following key parameters.

- `public_data.n_classes`: the number of categories for pretraining dataset (e.g., 365 for Places365).
- `public_data.name`: [`null`, `imagenet`, `places365`].
- `public_data.train_path`: the path to pretraining dataset.
- `public_data.selective.model_path` (need it?): .

We use ImageNet as the default pretraining dataset, and these parameters are configured accordingly.

For example,

(1) Using ImageNet to pretrain DPGAN using conditional pretraining.
```
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 --epsilon 10.0 eval.mode=val \
 public_data.name=imagenet \
 pretrain.cond=Ture \
 --exp_description pretrain_imagenet_conditional 
```

(2) Using Places365 to pretrain DPGAN using conditional pretraining.
```
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 --epsilon 10.0 eval.mode=val \
 public_data.name=places365 public_data.n_classes=365 public_data.train_path=dataset/places365 \
 pretrain.cond=Ture \
 --exp_description pretrain_places365_conditional 
```

(3) Using Places365 to pretrain DPGAN using unconditional pretraining.
```
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 --epsilon 10.0 eval.mode=val \
 public_data.name=places365 public_data.n_classes=365 public_data.train_path=dataset/places365 \
 pretrain.cond=False \
 --exp_description pretrain_places365_unconditional 
```

### 4.4 Results Explanation
We can find the `stdout.txt` files in the result folder, which record the training and evaluation processes. We explain the [file structure](./exp/README.md) of outputs in `exp`. After the evaluation, the results for each classifier training are available in `stdout.txt`.

In utility evaluation, after each classifier training, we can find,

```
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc of synthetic images on sensitive val and the corresponding acc on test dataset from wrn is 64.75999999999999 and 63.99
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc of synthetic images on noisy sensitive val and the corresponding acc on test dataset from wrn is 64.75999999999999 and 63.87
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc test dataset from wrn is 64.12
```
These results represent the best accuracy achieved by: (1) using the sensitive validation set (63.99%), (2) adding noise to the validation results of the sensitive dataset (`model.eval = val`), and the accuracy is 63.87%, and (3) using the sensitive test set for classifier selection (64.12%). 

If synthetic images are used as the validation set (`model.eval = syn`), the results after each classifier training would be:
```
INFO - evaluator.py - 2024-10-24 06:45:11,042 - The best acc of synthetic images on val (synthetic images) and the corresponding acc on test dataset from wrn is 63.175 and 56.22
INFO - evaluator.py - 2024-10-24 06:45:11,042 - The best acc test dataset from wrn is 64.22
```
These results present that the best accuracy achieved by: (1) using the synthetic images for validation set (56.22%) and (2) using the sensitive test set for classifier selection (64.22%).

The following results can be found at the end of the log file:
``` 
INFO - evaluator.py - 2024-11-13 21:19:44,813 - The best acc of accuracy (adding noise to the results on the sensitive set of validation set) of synthetic images from resnet, wrn, and resnext are [61.6, 64.36, 59.31999999999999].
INFO - evaluator.py - 2024-11-13 21:19:44,813 - The average and std of accuracy of synthetic images are 61.76 and 2.06
INFO - evaluator.py - 2024-11-13 21:50:27,195 - The FID of synthetic images is 21.644407353392182
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The Inception Score of synthetic images is 7.621163845062256
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The Precision and Recall of synthetic images is 0.5463906526565552 and 0.555840015411377
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The FLD of synthetic images is 7.258963584899902
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The ImageReward of synthetic images is -2.049745370597344
```
The first line shows the accuracy of the downstream task when noise is added to the validation results of the sensitive dataset for classifier selection (`model.eval = val`), across three studied classification outcomes. 

If synthetic images are used as the validation set (`model.eval = syn`), the first line would be:
```
INFO - evaluator.py - 2024-11-12 09:06:18,148 - The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are [59.48, 63.99, 59.53].
```
The synthetic images can be found at the `/exp/<algorithm_name>/<file_name>/gen/gen.npz`.

## 5. Customization

This part introduces how to apply DPImageBench for your own sensitive dataset.

## 6. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.


## Acknowledgement
 
 This project 
