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
│   └── mnist_28_eps1.0_LZN-2024-10-25-23-09-18/  
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
- `./pretrain/checkpoints/final_checkpoint.pth`:
- `./pretrain/checkpoints/snapshot_checkpoint.pth`:
- `./pretrain/samples/iter_2000`:
- `./train/checkpoints/final_checkpoint.pth`:
- `./train/checkpoints/snapshot_checkpoint.pth`:
- `./train/samples/iter_2000`:
- `./stdout.txt`: recording the training and evaluation results.

