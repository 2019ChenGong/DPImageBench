# DPImageBench

# Folder Description

`data` contains tools for data loading.
`DataLens` and `G-PATE` are methods to be implemented.
`dnnlib` and `torch_utils` are used for load Inception V3.
`models` contains the implemented methods.
`opacus` is a modified opacus package.
`exp` contains my debug logs, and does not need to be uploaded.

# Get Start on DPLab

 ```
conda activate dpdm
cd /p/fzv6enresearch/DPImageBench
python run.py configs/DP_MERF/mnist.yaml
 ```