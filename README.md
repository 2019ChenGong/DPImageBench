<div align=center>
  
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis Algorithms
</div>

## Todo

- [] recoding the intermediate results of methods with tqdm.

- [] the hyper-parameters of classification methods for fashionmnist are unreasonable.

- [] mnist for g-pate, the acc is so low. Besides, the g-pate is too slow; it seems like that algorithms dosen't use GPU.

- [] dpsda errors. [KC: error still exists?] [Chen; Right]

- [] the accuracy for cifar10 and 100 is low.


## 1. Contents
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
    - [2.1 Currently Supported Algorithms](#21-currently-supported-algorithms)
  - [3. Repo Contents](#3-repo-contents)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Inference](#34-inference)
  - [4. Contacts](#4-contacts)
  - [5. Acknowledgment](#5-acknowledgment)

## 2. Introduction

### 2.1 Currently Supported Algorithms

We list currently supported DP image synthesis methods as follows.

  | Methods |  Link                                                         |
  | -------------- | ------------------------------------------------------------ |
  | DP-MERF            |  [\[AISTATS 2021\] DP-MERF: Differentially Private Mean Embeddings With Randomfeatures for Practical Privacy-Preserving Data Generation](https://proceedings.mlr.press/v130/harder21a.html) |
  | DP-Kernel        |  [\[NeuriPS 2023\] Functional Renyi Differential Privacy for Generative Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html) |
  | DPSDA          |  [\[ICLR 2024\] Differentially Private Synthetic Data via Foundation Model {API}s 1: Images](https://openreview.net/forum?id=YEhQs8POIo) |
  | G-PATE            |  [\[NeuriPS 2021\] G-PATE: Scalable Differentially Private Data Generator via Private Aggregation of Teacher Discriminators](https://proceedings.neurips.cc/paper_files/paper/2021/hash/171ae1bbb81475eb96287dd78565b38b-Abstract.html) |
  | DataLens            |  [\[CCS 2021\] DataLens: Scalable Privacy Preserving Training via Gradient Compression and Aggregation](https://dl.acm.org/doi/abs/10.1145/3460120.3484579) |
  | DP-GAN            |  [\[1802.06739\] Differentially Private Generative Adversarial Network (arxiv.org)](https://arxiv.org/abs/1802.06739) |
  | DPDM          |  [\[TMLR 2023\] Differentially Private Diffusion Models](https://openreview.net/forum?id=ZPpQk7FJXF) |
  | PDP-Diffusion       | [\[2302.13861\] Differentially Private Diffusion Models Generate Useful Synthetic Images (arxiv.org)](https://arxiv.org/abs/2305.15759) |
  | DP-LDM            | [\[2305.15759\] Differentially Private Latent Diffusion Models (arxiv.org)](https://arxiv.org/abs/2302.13861)            |
  | DP-Promise       | [\[UESNIX Security 2024\] DP-promise: Differentially Private Diffusion Probabilistic Models for Image Synthesis](https://www.usenix.org/conference/usenixsecurity24/presentation/wang-haichen) |
  | PrivImage       | [\[UESNIX Security 2024\] PrivImage: Differentially Private Synthetic Image Generation using Diffusion Models with Semantic-Aware Pretraining](https://www.usenix.org/conference/usenixsecurity24/presentation/li-kecen) |


## 3. Repo Contents

Below is the directory structure of the DPImageBench project, which encapsulates its three core functionalities within the watermark/, visualize/, and evaluation/ directories. To facilitate user understanding and demonstrate the toolkit's ease of use, we provide a variety of test cases. The test code can be found in the test/ directory.


```plaintext
DPImageBench/
├── config/                     # Configuration files for various DP image synthesis algorithms
│   ├──            
│   ├── DP-MERF      
│   ├── DP-NTK       
│   ├── DP-Kernel
│   ├── DPSDA            
│   ├── G-PATE            
│   ├── DP-GAN         
│   ├── DPDM        
│   ├── PDP-Diffusion      
│   ├── DP-LDM   
│   ├── DP-Promise     
│   └── PDP-Diffusion         
├── dataset/                    # Datasets studied in the project
│   ├── camelyon/
│   ├── celeba/
│   ├── imagenet/
│   ├── mnist/
│   └── wmt16_de_en/
├── evaluation/                 # Evaluation module of MarkLLM, including tools and pipelines
│   ├── dataset.py              # Script for handling dataset operations within evaluations
│   ├── examples/               # Scripts for automated evaluations using pipelines
│   │   ├── assess_detectability.py  
│   │   ├── assess_quality.py    
│   │   └── assess_robustness.py   
│   ├── pipelines/              # Pipelines for structured evaluation processes
│   │   ├── detection.py    
│   │   └── quality_analysis.py 
│   └── tools/                  # Evaluation tools
│       ├── oracle.py
│       ├── success_rate_calculator.py  
        ├── text_editor.py         
│       └── text_quality_analyzer.py   
├── exceptions/                 # Custom exception definitions for error handling
│   └── exceptions.py
├── font/                       # Fonts needed for visualization purposes
├── MarkLLM_demo.ipynb          # Jupyter Notebook
├── test/                       # Test cases and examples for user testing
│   ├── test_method.py      
│   ├── test_pipeline.py    
│   └── test_visualize.py   
├── utils/                      # Helper classes and functions supporting various operations
│   ├── openai_utils.py     
│   ├── transformers_config.py 
│   └── utils.py            
├── visualize/                  # Visualization Solutions module of MarkLLM
│   ├── color_scheme.py    
│   ├── data_for_visualization.py  
│   ├── font_settings.py    
│   ├── legend_settings.py  
│   ├── page_layout_settings.py 
│   └── visualizer.py       
├── watermark/                  # Implementation framework for watermark algorithms
│   ├── auto_watermark.py       # AutoWatermark class
│   ├── base.py                 # Base classes and functions for watermarking
│   ├── ewd/                
│   ├── exp/               
│   ├── exp_edit/          
│   ├── kgw/
│   ├── its_edit/                 
│   ├── sir/               
│   ├── sweet/              
│   ├── unigram/           
│   ├── upv/                
│   └── xsir/               
├── README.md                   # Main project documentation
└── requirements.txt            # Dependencies required for the project
```

## 3. Get Start on DPImageBench

### 3.1 Install DPImageBench

 ```
conda create -n dpimagebench python=3.7
conda activate dpimagebench
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorflow-gpu==1.14.0
pip install requirements.txt
cd opacus; pip install -e .; cd ..
cd models/DPSDA/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..
cd models; gdown https://drive.google.com/uc?id=1yVTWzaSqJVDJy8CsZKtqDoBNeM6154D4; unzip pretrained_models.zip; cd ..
 ```

### 3.2 Prepare Dataset

 ```
mkdir dataset
sh scripts/download_dataset.sh
python preprocess_dataset.py; cd ..
 ```

### 3.3 Running

 ```
conda activate dpimagebench
cd DPImageBench
python run.py --config configs/{model_name}/{dataset_name}_eps{epsilon}.yaml
 ```

Available `model_name` are [`DP-NTK`, `DP-Kernel`, `DP-LDM`, `DP-MERF`, `DP-Promise`, `DPDM`, `DPSDA`, `G-PATE`, `PDP-Diffusion`, `PrivImage`].

Available `epsilon` is [`1.0`].

Available `dataset_name` is [`mnist_28`].

So far, I have only implemented FID in our evaluation.
