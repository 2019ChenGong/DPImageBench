import os, sys, argparse
from packaging import version
import logging
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision

import subprocess
from concurrent.futures import ProcessPoolExecutor

from models.DP_Diffusion.utils.util import make_dir
from models.synthesizer import DPSynther

def execute(script):
    # Replace Python's `None` with JSON's `null` in the script items.
    # If an item is `None`, it will be replaced with the string 'null'.
    script = ['null' if item is None else item.replace('None', 'null') for item in script]
    
    try:
        # Define the path to the Python interpreter.
        python_path = 'python'
        
        # Run the script using the Python interpreter.
        result = subprocess.run([python_path] + script, check=True, text=True, capture_output=True)
        
        # Return the standard output of the command.
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error message from stderr.
        print(f"error: {e.stderr}")
        
        # Return the error message from stderr.
        return e.stderr

class DP_LORA(DPSynther):
    def __init__(self, config, device):
        """
        Initializes the class instance with configuration and device settings.
        
        Args:
            config (Config): Configuration object containing setup and model parameters.
            device (str): Device on which the model will be run.
        """
        super().__init__()  # Call the initializer of the parent class
        
        self.local_rank = config.setup.local_rank  # Local rank for distributed training
        self.global_rank = config.setup.global_rank  # Global rank for distributed training
        self.global_size = config.setup.global_size  # Total number of processes in distributed training

        self.config = config  # Store the configuration object
        self.device = 'cuda:%d' % self.local_rank  # Set the device to use based on local rank

        self.private_num_classes = config.model.private_num_classes  # Number of private classes
        self.public_num_classes = config.model.public_num_classes  # Number of public classes
        self.label_dim = max(self.private_num_classes, self.public_num_classes)  # Dimension of the label space

        self.is_pretrain = True  # Flag to indicate if the model is in pre-training mode
    
    def pretrain(self, public_dataloader, config):
        # Check if the dataloader is provided; if not, set pretraining flag to False and return
        if public_dataloader is None:
            self.is_pretrain = False
            return
        
        # If this is the main process (rank 0), create the necessary log directory
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        # Pretrain the autoencoder using the public dataset
        self.pretrain_autoencoder(public_dataloader.dataset, config.autoencoder, os.path.join(config.log_dir, 'autoencoder'))
        
        # Pretrain the U-Net using the public dataset
        self.pretrain_unet(public_dataloader.dataset, config.unet, os.path.join(config.log_dir, 'unet'))

        # Clear the GPU cache to free up memory
        torch.cuda.empty_cache()

    def pretrain_autoencoder(self, public_dataset, config, logdir):
        # If a checkpoint is already available, skip pretraining
        if self.config.model.ckpt is not None:
            return
        
        # If this is the main process (rank 0), create the log directory
        if self.global_rank == 0:
            make_dir(logdir)

        # Determine the visible GPUs
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or True:
            gpu_ids = ','.join([str(i) for i in range(self.config.setup.n_gpus_per_node)]) + ','
        else:
            gpu_ids = str(cuda_visible_devices) + ','

        # Set the path to the configuration file
        config_path = config.config_path

        # Determine the data target based on the training path
        if 'imagenet' in self.config.public_data.train_path:
            data_target = 'data.SpecificImagenet.SpecificClassImagenet'
        elif 'places' in self.config.public_data.train_path:
            data_target = 'data.SpecificPlaces365.SpecificClassPlaces365_ldm'

        # Prepare the command-line arguments for the pretraining script
        scripts = [[
            'models/DP_LDM/main.py', 
            '-t', 
            '--logdir', logdir, 
            '--base', config_path, 
            '--gpus', gpu_ids, 
            'model.params.output_file={}'.format(os.path.join(os.path.dirname(os.path.dirname(logdir)), 'stdout.txt')),
            'data.params.batch_size={}'.format(config.batch_size), 
            'lightning.trainer.max_epochs={}'.format(config.n_epochs), 
            'data.params.train.target={}'.format(data_target),
            'data.params.validation.target={}'.format(data_target),
            'data.params.train.params.root={}'.format(self.config.public_data.train_path),
            'data.params.validation.params.root={}'.format(self.config.public_data.train_path),
            'data.params.train.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.validation.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.train.params.c={}'.format(self.config.public_data.num_channels),
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.train.data_num={}'.format(len(public_dataset)),
            'data.params.validation.data_num={}'.format(len(public_dataset)),
        ]]

        # Execute the pretraining script using a process pool
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"Generated an exception: {e}")

        # Update the model checkpoint path after pretraining
        self.config.model.ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')

    def pretrain_unet(self, public_dataset, config, logdir):
        # If this is the main process (rank 0), create the log directory
        if self.global_rank == 0:
            make_dir(logdir)

        # Determine the visible GPUs
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or True:
            gpu_ids = ','.join([str(i) for i in range(self.config.setup.n_gpus_per_node)]) + ','
        else:
            gpu_ids = str(cuda_visible_devices) + ','

        # Set the path to the configuration file
        config_path = config.config_path

        # Determine the pretraining model path and the model target
        pretrain_model = self.config.model.ckpt
        if 'unet' in pretrain_model:
            model_target = 'model.params.ckpt_path='
        else:
            model_target = 'model.params.first_stage_config.params.ckpt_path='

        # Determine the data target based on the training path
        if 'imagenet' in self.config.public_data.train_path:
            data_target = 'data.SpecificImagenet.SpecificClassImagenet_ldm'
        elif 'places' in self.config.public_data.train_path:
            data_target = 'data.SpecificPlaces365.SpecificClassPlaces365_ldm'

        # Determine if a specific class is selected
        if self.config.public_data.selective.ratio == 1.0:
            specific_class = None
        else:
            specific_class = self.config.public_data.selective.semantic_path

        # Prepare the command-line arguments for the pretraining script
        scripts = [[
            'models/DP_LDM/main.py', 
            '-t', 
            '--logdir', logdir, 
            '--base', config_path, 
            '--gpus', gpu_ids, 
            'model.params.cond_stage_config.params.n_classes={}'.format(self.label_dim + 1),
            'data.params.cond={}'.format(self.config.pretrain.cond),
            'data.params.train.target={}'.format(data_target),
            'data.params.validation.target={}'.format(data_target),
            'model.params.output_file={}'.format(os.path.join(os.path.dirname(os.path.dirname(logdir)), 'stdout.txt')),
            'model.params.unet_config.params.attention_resolutions={}'.format([2**i for i in range(len(self.config.model.network.attn_resolutions))]),
            'model.params.unet_config.params.channel_mult={}'.format(self.config.model.network.ch_mult),
            'model.params.unet_config.params.model_channels={}'.format(self.config.model.network.nf),
            'data.params.batch_size={}'.format(config.batch_size), 
            'lightning.trainer.max_epochs={}'.format(config.n_epochs), 
            model_target + str(pretrain_model),
            'data.params.train.params.root={}'.format(self.config.public_data.train_path),
            'data.params.validation.params.root={}'.format(self.config.public_data.train_path),
            'data.params.train.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.validation.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.train.params.c={}'.format(self.config.public_data.num_channels),
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.train.data_num={}'.format(len(public_dataset)),
            'data.params.validation.data_num={}'.format(len(public_dataset)),
            'data.params.train.params.specific_class={}'.format(specific_class),
            'data.params.validation.params.specific_class={}'.format(specific_class),
        ]]

        # Execute the pretraining script using a process pool
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"Generated an exception: {e}")

        # Update the model checkpoint path after pretraining
        self.config.model.ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')

    def train(self, sensitive_dataloader, config):
        """
        Train the model using the provided sensitive data loader and configuration.

        Parameters:
        - sensitive_dataloader: DataLoader for the sensitive dataset.
        - config: Configuration object containing training parameters and paths.
        """
        if sensitive_dataloader is None or config.n_epochs == 0:
            # If the dataloader is None or the number of epochs is zero, exit early.
            return
        
        if self.global_rank == 0:
            # Create the log directory if this is the main process (rank 0).
            make_dir(config.log_dir)
        
        # Define GPU IDs to be used.
        gpu_ids = '0,'
        # Path to the configuration file.
        config_path = config.config_path
        # Path to the pretrained model checkpoint.
        pretrain_model = self.config.model.ckpt

        # Construct the command-line arguments for the training script.
        scripts = [[
            'models/DP_LORA/main.py',  # Path to the main training script.
            '-t',  # Training flag.
            '--logdir', config.log_dir,  # Directory to save logs.
            '--base', config_path,  # Base configuration file.
            '--gpus', gpu_ids,  # GPUs to use.
            '--accelerator', 'gpu',  # Accelerator type (GPU in this case).
            'model.params.cond_stage_config.params.n_classes={}'.format(self.label_dim + 1),  # Number of classes.
            'model.params.unet_config.params.attention_resolutions={}'.format([2**i for i in range(len(self.config.model.network.attn_resolutions))]),  # Attention resolutions.
            'model.params.unet_config.params.channel_mult={}'.format(self.config.model.network.ch_mult),  # Channel multiplier.
            'model.params.unet_config.params.model_channels={}'.format(self.config.model.network.nf),  # Model channels.
            'model.params.output_file={}'.format(os.path.join(os.path.dirname(config.log_dir), 'stdout.txt')),  # Output file for logs.
            'model.params.ckpt_path={}'.format(pretrain_model),  # Pretrained model checkpoint path.
            'model.params.dp_config.epsilon={}'.format(config.dp.epsilon),  # Differential privacy epsilon.
            'model.params.dp_config.delta={}'.format(config.dp.delta),  # Differential privacy delta.
            'model.params.dp_config.max_grad_norm={}'.format(config.dp.max_grad_norm),  # Maximum gradient norm.
            'model.params.dp_config.max_batch_size={}'.format(config.batch_size // config.n_splits),  # Maximum batch size per split.
            'data.params.batch_size={}'.format(config.batch_size),  # Batch size.
            'lightning.trainer.max_epochs={}'.format(config.n_epochs),  # Number of epochs.
            'data.params.train.params.path={}'.format(self.config.sensitive_data.train_path),  # Path to the training data.
            'data.params.validation.params.path={}'.format(self.config.sensitive_data.train_path),  # Path to the validation data.
            'data.params.train.params.resolution={}'.format(self.config.sensitive_data.resolution),  # Resolution of the training data.
            'data.params.validation.params.resolution={}'.format(self.config.sensitive_data.resolution),  # Resolution of the validation data.
            'data.params.train.params.c={}'.format(self.config.sensitive_data.num_channels),  # Number of channels in the training data.
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),  # Number of channels in the validation data.
            'data.params.train.data_num={}'.format(len(sensitive_dataloader.dataset)),  # Number of training samples.
            'data.params.validation.data_num={}'.format(len(sensitive_dataloader.dataset)),  # Number of validation samples.
        ]]

        # Use a process pool to execute the training script.
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")  # Log the output of the script.
                except Exception as e:
                    logging.info(f"Generated an exception: {e}")  # Log any exceptions that occur.

        # Update the checkpoint path in the configuration.
        self.config.model.ckpt = os.path.join(self.config.train.log_dir, 'checkpoints', 'last.ckpt')


    def generate(self, config):
        # Log the start of the generation process and the number of samples to be generated
        logging.info("start to generate {} samples".format(config.data_num))
        
        # If this is the main process (rank 0) and the log directory does not exist, create it
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        # Define the command-line arguments for the script that will be executed
        scripts = [[
            'models/DP_LORA/cond_sampling_test.py',  # Path to the script
            '--save_path', config.log_dir,  # Directory where the generated data will be saved
            '--yaml', self.config.train.config_path,  # Path to the configuration file
            '--ckpt_path', self.config.model.ckpt,  # Path to the model checkpoint
            '--num_samples', str(config.data_num),  # Number of samples to generate
            '--num_classes', str(self.config.sensitive_data.n_classes),  # Number of classes in the dataset
            '--batch_size', str(config.batch_size),  # Batch size for generation
            'model.params.unet_config.params.attention_resolutions={}'.format([2**i for i in range(len(self.config.model.network.attn_resolutions))]),  # Attention resolutions for the UNet model
            'model.params.unet_config.params.channel_mult={}'.format(self.config.model.network.ch_mult),  # Channel multiplier for the UNet model
            'model.params.unet_config.params.model_channels={}'.format(self.config.model.network.nf),  # Number of initial channels for the UNet model
            'model.params.cond_stage_config.params.n_classes={}'.format(self.label_dim + 1),  # Number of classes plus one for the conditional stage.
        ]]
        
        # Use a ProcessPoolExecutor to run the script in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]  # Submit the script execution tasks
            for future in futures:
                try:
                    output = future.result()  # Get the result of the task
                    logging.info(f"Output:\n{output}")  # Log the output
                except Exception as e:
                    logging.info(f"generated an exception: {e}")  # Log any exceptions
        
        # Log the completion of the generation process
        logging.info("Generation Finished!")
        
        # Load the generated synthetic data from the specified directory
        syn = np.load(os.path.join(config.log_dir, 'gen.npz'))
        syn_data, syn_labels = syn["x"], syn["y"]  # Extract the data and labels
        
        # Prepare a list of images to display, taking the first 8 images from each class
        show_images = []
        for cls in range(self.config.sensitive_data.n_classes):
            show_images.append(syn_data[syn_labels == cls][:8])
        show_images = np.concatenate(show_images)
        
        # Save the selected images to a PNG file
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        
        # Return the synthetic data and labels
        return syn_data, syn_labels
