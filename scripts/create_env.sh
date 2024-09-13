conda create -n dpbench_test python=3.7
conda activate dpbench_test
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip instal tensorflow-gpu==1.14.0
pip install re.txt
cd opacus; pip install -e .; cd ..
cd models/DPSDA/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..
