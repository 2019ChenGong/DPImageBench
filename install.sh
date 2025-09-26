#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

conda create -n dpimagebench python=3.9 -y
conda activate dpimagebench;
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121;
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0;
pip install -r requirements.txt;
cd opacus; pip install -e .; cd ..
cd models/PE; pip install improved-diffusion@git+https://github.com/fjxmlzn/improved-diffusion.git@8f6677c3c47d1c1ad2e22ad2603eaec4cc639805; cd ..; cd ..
cd models/PE; pip install git+https://github.com/openai/guided-diffusion.git; cd ..; cd ..
cd models/DP_LORA/peft; pip install -e .; cd ..; cd ..; cd ..
pip install transformers==4.27.4
cd models; gdown https://drive.google.com/uc?id=1yVTWzaSqJVDJy8CsZKtqDoBNeM6154D4; unzip pretrained_models.zip; cd ..