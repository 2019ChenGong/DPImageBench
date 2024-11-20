# Results Structure

We outline the structure of the results files as follows. The training and evaluations results are recorded in the file `exp`. For example, if users leverage the PDP-Diffusion method to generate synthetic images for the MNIST dataset under a privacy budget of `eps=1.0`, the structure of the folder is as follows:

```plaintext
exp/                                  
├── dp-kernel/                              
├── dp-ldm/ 
├── dp-merf/
├── dp-ntk/ 
├── dpdm/ 
├── dpgan/ 
├── gs-wgan/ 
├── pdp-diffusion/ 
│   └── mnist_28_eps1.0-2024-10-25-23-09-18/  
│           ├── gen  
│           │   ├── gen.npz 
│           │   └── sample.png 
│           ├── pretrain  
│           │   ├── checkpoints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth  
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── train
│           │   ├── checkooints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth    
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           └──stdout.txt   
├── pe/ 
└── privimage/  
```

We introduce the files as follows,

- `./gen/gen.npz`: the synthetic images.
- `./gen/sample.png`: the samples of synthetic images.
- `./pretrain/checkpoints/final_checkpoint.pth`: the parameters of synthsizer at the final epochs.
- `./pretrain/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./pretrain/samples/iter_2000`: the synthetic images under 2000 iterations for pretraining on public datasets.
- `./train/checkpoints/final_checkpoint.pth`: the parameters of synthsizer at the final epochs.
- `./train/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./train/samples/iter_2000`: the synthetic images under 2000 iterations for training on sensitive datasets.
- `./stdout.txt`: the file used to record the training and evaluation results.

