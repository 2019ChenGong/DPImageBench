<div align=center>
  
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis Algorithms
</div>

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

## Updates 
- 🎉 **(2024.11.19)** We're thrilled to announce the release of initial version of DPImageBench!

## Todo

- [ ] recoding the intermediate results of methods with tqdm.

- [ ] setup.master_port=6026

- [ ] keep consistent cond and uncond for dm and gan

- [ ] GSWGAN merge

- [ ] use a bash to represent installation.

- [ ] End to end implementation for PrivImage

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


## 3. Repo Contents

Below is the directory structure of the DPImageBench project, which encapsulates its three core functionalities within the watermark/, visualize/, and evaluation/ directories. To facilitate user understanding and demonstrate the toolkit's ease of use, we provide a variety of test cases. The test code can be found in the test/ directory.


```plaintext
DPImageBench/
├── config/                     # Configuration files for various DP image synthesis algorithms
│   ├──            
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
cd models/PE/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..
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

Available `model_name` are [`DP-NTK`, `DP-Kernel`, `DP-LDM`, `DP-MERF`, `DP-Promise`, `DPDM`, `PE`, `G-PATE`, `PDP-Diffusion`, `PrivImage`].

Available `epsilon` is [`1.0`].

Available `dataset_name` is [`mnist_28`].

So far, I have only implemented FID in our evaluation.
