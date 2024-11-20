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
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Inference](#34-inference)
  - [5. Customization](#5-customization)
  - [6. Contacts](#6-contacts)
  - [Acknowledgment](#acknowledgement)

### Updates 
- 🎉 **(2024.11.19)** We're thrilled to announce the release of initial version of DPImageBench!

### Todo
- [ ] recoding the intermediate results of methods with tqdm.

- [ ] setup.master_port=6026

- [ ] keep consistent cond and uncond for dm and gan

- [ ] use a bash to represent installation.

- [ ] End to end implementation for PrivImage

- [ ] remove the unneccessary part for algorithms in models

- [x] using 'val' as the default setting but use 'sen' to represent the original evaluation method, it seems like using 'sen' as the default setting now. [KC: All configs have a new attribute "eval.mode" with default value "val"]

- [x] End to end implementation of data preparation. There two problems (1) lack of downloading for places365; (2) it seems like should run preprocess_dataset.py based on the sensitive dataset one by one. Can we just use one instruction? [KC: (1) downloading places365 is included in preprocess_dataset.py (2) by default, preprocess_dataset.py downloads and processes all needed datasets when --data_name is not specified.]

- [ ] Customize privacy budget

- [ ] Checkpoint?

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
  | PDP-Diffusion       | [\[2302.13861\] Differentially Private Diffusion Models Generate Useful Synthetic Images (arxiv.org)](https://arxiv.org/abs/2305.15759) |
  | DP-LDM            | [\[TMLR 2024\] Differentially Private Latent Diffusion Models (arxiv.org)](https://arxiv.org/abs/2302.13861)            |
  | PrivImage       | [\[UESNIX Security 2024\] PrivImage: Differentially Private Synthetic Image Generation using Diffusion Models with Semantic-Aware Pretraining](https://www.usenix.org/conference/usenixsecurity24/presentation/li-kecen) |

### 2.2 Currently Supported Datasets
We list the studied datasets as follows, which include seven sensitive datasets and two public datasets.
  | Dataset | Usage   |
  | ------- | --------------------- |
  | ImageNet_ILSVRC2012             |  Pretraining dataset  |
  | Places365             |  Pretraining dataset  |
  | MNIST             |  Sensitive dataset  |
  | FashionMNIST             |  Sensitive dataset  |
  | CIFAR-10                   |  Sensitive dataset  |      
  | CIFAR-100                   |  Sensitive dataset  |      
  | EuroSAT             |  Sensitive dataset  |
  | CelebA             |  Sensitive dataset  |
  | Camelyon                   |  Sensitive dataset  |                                     |

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
sh scripts/install.sh
 ```

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

### 4.3 Prepare Pretrained Models

### 4.3 Running

#### 4.3.1 Key hyper-parameter introductions.

We list the key hyper-parameters below, including their explanations and available options.

Available `model_name` are [`DP-NTK`, `DP-Kernel`, `DP-LDM`, `DP-MERF`, `DP-Promise`, `DPDM`, `PE`, `G-PATE`, `PDP-Diffusion`, `PrivImage`].

Available `epsilon` is [`1.0`, `10.0`].

- `--dataset_name`: means the sensitive dataset, the option is [`mnist_28`, `fmnist_28`, `cifar10_32`, `cifar100_32`, `eurosat_32`, `celeba_male_32`, `camelyon_32`].
- `setup.n_gpus_per_node`: means the number of GPUs to be used for training.

#### 4.3.2 How to run.

```
conda activate dpimagebench
cd DPImageBench
python run.py --config configs/{model_name}/{dataset_name}_eps{epsilon}.yaml
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

If synthetic images are used as the validation set (`model.eval = sen`), the results after each classifier training would be:
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

If synthetic images are used as the validation set (`model.eval = sen`), the first line would be:
```
INFO - evaluator.py - 2024-11-12 09:06:18,148 - The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are [59.48, 63.99, 59.53000000000001].
```
The synthetic images can be found at the `/exp/<algorithm_name>/<file_name>/gen/gen.npz`.

## 5. Customization

## 6. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.


## Acknowledgement
 
