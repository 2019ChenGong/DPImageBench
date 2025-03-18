mkdir dataset;
cd dataset;
mkdir celeba; cd celeba; gdown https://drive.google.com/uc?id=1nISkV4NyL9Ntd9E_A8dU2YpvO6Pcw8lx; unzip celeba.zip; cd ..
mkdir camelyon; cd camelyon; gdown https://drive.google.com/uc?id=1cqjFYIAkERAWapbI6rmf_F9x9IBfz8Gl; unzip camelyon_32.zip; cd ..
mkdir imagenet; cd imagenet; gdown https://drive.google.com/uc?id=1SFvDfBWmG30xTjFJ0v9Em5avRVL6yqAh; unzip imagenet_32.zip; cd ..
mkdir eurosat; cd eurosat; wget -c https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip; unzip EuroSAT_RGB.zip; cd ..
cd ..; cd data;
python preprocess_dataset.py; cd ..