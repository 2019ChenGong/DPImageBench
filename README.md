<div align=center>
  
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis Algorithms
</div>

## 1. Contents
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Inference](#34-inference)
  - [4. Contacts](#4-contacts)
  - [5. Acknowledgment](#5-acknowledgment)

```plaintext
MarkLLM/
├── config/                     # Configuration files for various watermark algorithms
│   ├── EWD.json           
│   ├── EXPEdit.json       
│   ├── EXP.json           
│   ├── KGW.json
│   ├── ITSEdit.json            
│   ├── SIR.json            
│   ├── SWEET.json         
│   ├── Unigram.json        
│   ├── UPV.json           
│   └── XSIR.json           
├── dataset/                    # Datasets used in the project
│   ├── c4/
│   ├── human_eval/
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

# Folder Description

`data` contains tools for data loading.

`DataLens` and `G-PATE` are methods to be implemented.

`dnnlib` and `torch_utils` are used for load Inception V3.

`models` contains the implemented methods.

`opacus` is a modified opacus package.

`exp` contains my debug logs, and does not need to be uploaded.

# Get Start on DPImageBench

 ```
conda activate dpdm
cd /p/fzv6enresearch/DPImageBench
python run.py configs/DP_MERF/mnist.yaml
 ```