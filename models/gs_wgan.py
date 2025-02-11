import os
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import random
import numpy as np
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier

from models.GS_WGAN.models_ import *
from models.GS_WGAN.utils import *
from models.GS_WGAN.ops import exp_mov_avg
from models.DP_GAN.generator import Generator

from models.synthesizer import DPSynther

def warm_up(script):
    try:
        result = subprocess.run(['python'] + script, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"error: {e.stderr}")
        return e.stderr

class GS_WGAN(DPSynther):
    def __init__(self, config, device):
        # Call the superclass constructor
        super().__init__()

        # Initialize configuration parameters from the provided config object
        self.num_discriminators = config.num_discriminators  # Number of discriminators to be used
        self.z_dim = config.Generator.z_dim  # Dimension of the latent space (z) for the generator
        self.c = config.c  # Number of channels in the input images
        self.img_size = config.img_size  # Size of the input images
        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        self.label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the label dimension based on the maximum number of classes
        self.latent_type = config.latent_type  # Type of latent variable used
        self.ckpt = config.ckpt  # Path to checkpoint file for model loading

        # Initialize the generator network
        self.netG = GeneratorResNet(
            c=self.c, 
            img_size=self.img_size, 
            z_dim=self.z_dim, 
            model_dim=config.Generator.g_conv_dim, 
            num_classes=self.label_dim
        )

        # Create a deep copy of the generator network for shadow training or other purposes
        self.netGS = copy.deepcopy(self.netG)

        # Initialize a list to hold multiple discriminator networks
        self.netD_list = []
        for i in range(self.num_discriminators):
            netD = DiscriminatorDCGAN(
                c=self.c, 
                img_size=self.img_size, 
                num_classes=self.private_num_classes
            )
            self.netD_list.append(netD)  # Add each discriminator to the list

        # Calculate and log the number of trainable parameters in the generator network
        model_parameters = filter(lambda p: p.requires_grad, self.netG.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)

        # Store the configuration and device information
        self.config = config
        self.device = device

    
    def pretrain(self, public_dataloader, config):
        # Check if the dataloader is provided, if not, return immediately
        if public_dataloader is None:
            return
        
        # Create a directory to store logs
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))
        os.mkdir(os.path.join(config.log_dir, "samples"))

        # Set seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Generate fixed noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError("Unsupported latent type")

        # Move networks to the device (CPU/GPU)
        netG = self.netG.to(self.device)
        netGS = self.netGS.to(self.device)

        # Initialize the discriminator and optimizers
        netD = DiscriminatorDCGAN(c=self.c, img_size=self.img_size, num_classes=self.private_num_classes).to(self.device)
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        # Register a backward hook to the first convolutional layer of the discriminator
        global dynamic_hook_function
        netD.conv1.register_backward_hook(master_hook_adder)

        # Infinite data generator from the public dataset
        input_data = inf_train_gen(public_dataloader)

        # Training loop
        for iter in range(config.iterations + 1):

            # Enable gradient computation for the discriminator parameters
            for p in netD.parameters():
                p.requires_grad = True

            # Train the discriminator multiple times per generator update
            for iter_d in range(config.critic_iters):
                real_data, real_y = next(input_data)
                if not config.cond:
                    real_y = torch.zeros_like(real_y).long()
                real_y = real_y % self.private_num_classes
                batchsize = real_data.shape[0]
                real_data = real_data.view(batchsize, -1)
                real_data = real_data.to(self.device)
                real_y = real_y.to(self.device)
                real_data_v = autograd.Variable(real_data)

                # Set the dynamic hook function to a dummy function
                dynamic_hook_function = dummy_hook
                netD.zero_grad()
                D_real_score = netD(real_data_v, real_y)
                D_real = -D_real_score.mean()

                # Generate noise based on the latent type
                if self.latent_type == 'normal':
                    noise = torch.randn(batchsize, self.z_dim).to(self.device)
                elif self.latent_type == 'bernoulli':
                    noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
                else:
                    raise NotImplementedError("Unsupported latent type")
                noisev = autograd.Variable(noise)
                fake = autograd.Variable(netG(noisev, real_y.to(self.device)).view(batchsize, -1).data)
                inputv = fake.to(self.device)
                D_fake = netD(inputv, real_y.to(self.device))
                D_fake = D_fake.mean()

                # Compute the gradient penalty
                gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, config.L_gp, self.device)
                D_cost = D_fake + D_real + gradient_penalty

                # Add a regularization term to the loss
                logit_cost = config.L_epsilon * torch.pow(D_real_score, 2).mean()
                D_cost += logit_cost

                # Backpropagate and update the discriminator
                D_cost.backward()
                Wasserstein_D = -D_real - D_fake
                optimizerD.step()

            # Clean up variables to free memory
            del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
            torch.cuda.empty_cache()

            # Set the dynamic hook function to modify gradients
            dynamic_hook_function = modify_gradnorm_conv_hook

            # Disable gradient computation for the discriminator parameters
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            # Generate noise and labels for the generator
            if self.latent_type == 'normal':
                noise = torch.randn(batchsize, self.z_dim).to(self.device)
            elif self.latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
            else:
                raise NotImplementedError("Unsupported latent type")
            label = torch.randint(0, self.private_num_classes, [batchsize]).to(self.device)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, label).view(batchsize, -1)
            fake = fake.to(self.device)
            label = label.to(self.device)
            G = netD(fake, label)
            G = - G.mean()

            # Backpropagate and update the generator
            G.backward()
            G_cost = G
            optimizerG.step()

            # Apply exponential moving average to the generator's parameters
            exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

            # Print and log the training progress
            if iter < 5 or iter % config.print_step == 0:
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                    D_cost.cpu().data,
                                                                    Wasserstein_D.cpu().data
                                                                    ))
                logging.info('Step: {}, G_cost:{}, D_cost:{}, Wasserstein:{}'.format(iter, G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))

            # Generate images and save them periodically
            if iter % config.vis_step == 0:
                generate_image(iter, netGS, fix_noise, os.path.join(config.log_dir, "samples"), self.device, c=self.c, img_size=self.img_size, num_classes=self.private_num_classes)

            # Clean up variables to free memory
            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        # Save the trained models
        torch.save(self.netG.state_dict(), os.path.join(config.log_dir, "checkponts", 'netG.pth'))
        torch.save(self.netGS.state_dict(), os.path.join(config.log_dir, "checkponts", 'netGS.pth'))

    
    def warmup_training(self, config):
        # Number of GPUs to use for training
        n_gpu = config.n_gpu
        
        # Number of iterations for the training process
        iters = str(config.iterations)
        
        # Name of the dataset being used
        data_name = config.data_name
        
        # Number of samples to use for training (evaluation mode)
        train_num = config.eval_mode
        
        # Path to the dataset
        data_path = config.data_path
        
        # Architecture of the generator model
        gen_arch = "ResNet"
        
        # Size of the images used in training
        img_size = str(self.img_size)
        
        # Number of channels in the input images
        c = str(self.c)
        
        # Directory where logs will be saved
        log_dir = config.log_dir
        
        # Number of discriminators to use
        ndis = str(self.num_discriminators)
        
        # Number of discriminators per job
        dis_per_job = 250
        
        # Calculate the number of jobs needed based on the number of discriminators and discriminators per job
        njobs = self.num_discriminators // dis_per_job
        
        # List to hold the command-line scripts for each job
        scripts = []
        
        # Get the visible GPU IDs from the environment variable or default to all available GPUs
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = [i for i in range(n_gpu)]
        else:
            gpu_ids = cuda_visible_devices.split(',')
        
        # Loop over each GPU ID
        for gpu_id in range(len(gpu_ids)):
            gpu = gpu_ids[gpu_id]
            
            # Calculate the starting index for the meta discriminator
            meta_start = dis_per_job // n_gpu * gpu_id
            
            # Loop over each job
            for job_id in range(njobs):
                # Calculate the start and end indices for the current job's discriminators
                start = (job_id * dis_per_job + meta_start)
                end = (start + dis_per_job)
                
                # Generate a list of discriminator IDs for the current job
                vals = [str(dis_id) for dis_id in range(start, end)]
                
                # Construct the command-line script for the current job
                script = [
                    'models/GS_WGAN/pretrain.py', 
                    '-data', data_name, 
                    '--log_dir', log_dir, 
                    '--train_num', train_num, 
                    '-ndis', ndis, 
                    '-ids'] + vals + [
                    '--img_size', img_size, 
                    '--c', c, 
                    '--private_num_classes', str(self.private_num_classes), 
                    '--public_num_classes', str(self.public_num_classes), 
                    '--gpu_id', str(gpu), 
                    '--data_path', data_path, 
                    '-piters', iters, 
                    '--gen_arch', gen_arch, 
                    '--z_dim', str(self.z_dim), 
                    '--latent_type', self.latent_type, 
                    '--model_dim', str(self.config.Generator.g_conv_dim)
                ]
                scripts.append(script)
        
        # Add a final script to check if the training has finished
        scripts.append([
            'models/GS_WGAN/is_finished.py', 
            '--D_path', log_dir, 
            '--output_path', os.path.join(os.path.dirname(os.path.dirname(log_dir)), 'stdout.txt'), 
            '--D_num', ndis
        ])
        
        # Use a process pool to run the scripts in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(warm_up, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                except Exception as e:
                    logging.info(f"generated an exception: {e}")


    def train(self, sensitive_dataloader, config):
        # Check if the dataloader is provided
        if sensitive_dataloader is None:
            return
        
        # Create the log directory if it doesn't exist
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))
        os.mkdir(os.path.join(config.log_dir, "samples"))

        # Initialize the model based on whether a checkpoint is provided
        if self.ckpt is None:
            config.pretrain.log_dir = os.path.join(config.log_dir, 'warm_up')
            os.mkdir(config.pretrain.log_dir)
            indices_full = np.arange(len(sensitive_dataloader.dataset))
            np.random.shuffle(indices_full)
            np.save(os.path.join(config.pretrain.log_dir, 'indices.npy'), indices_full)
            self.warmup_training(config.pretrain)
            load_dir = config.pretrain.log_dir
        else:
            load_dir = self.ckpt
            indices_full = np.load(os.path.join(load_dir, 'indices.npy'), allow_pickle=True)
        
        # Ensure the indices match the dataset size
        if len(indices_full) != len(sensitive_dataloader.dataset):
            indices_full = np.arange(len(sensitive_dataloader.dataset))
            np.random.shuffle(indices_full)
            np.save(os.path.join(config.log_dir, 'indices.npy'), indices_full)
        
        # Calculate the size of the training set for each discriminator
        trainset_size = int(len(sensitive_dataloader.dataset) / self.num_discriminators)
        logging.info('Size of the dataset: {}'.format(trainset_size))

        # Calculate the noise multiplier for differential privacy
        self.noise_factor = get_noise_multiplier(
            target_epsilon=config.dp.epsilon, 
            target_delta=config.dp.delta, 
            sample_rate=1. / self.num_discriminators, 
            steps=config.iterations
        )
        global noise_multiplier
        noise_multiplier = self.noise_factor
        logging.info("The noise factor is {}".format(self.noise_factor))

        # Set seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Generate fixed noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError

        # Move models to the specified device
        netG = self.netG.to(self.device)
        netGS = self.netGS.to(self.device)
        for netD_id, netD in enumerate(self.netD_list):
            self.netD_list[netD_id] = netD.to(self.device)

        # Initialize optimizers for discriminators and generator
        optimizerD_list = []
        for i in range(self.num_discriminators):
            netD = self.netD_list[i]
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD.load_state_dict(torch.load(network_path))
            optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
            optimizerD_list.append(optimizerD)
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        # Create data loaders for each discriminator
        input_pipelines = []
        dataset = sensitive_dataloader.dataset
        for i in range(self.num_discriminators):
            start = i * trainset_size
            end = (i + 1) * trainset_size
            indices = indices_full[start:end]
            trainloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=config.batch_size, 
                drop_last=False, 
                sampler=SubsetRandomSampler(indices)
            )
            input_data = inf_train_gen(trainloader)
            input_pipelines.append(input_data)

        # Register backward hooks for discriminators
        global dynamic_hook_function
        for netD in self.netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)

        # Training loop
        for iter in range(config.iterations + 1):
            # Select a random discriminator
            netD_id = np.random.randint(self.num_discriminators, size=1)[0]
            netD = self.netD_list[netD_id]
            optimizerD = optimizerD_list[netD_id]
            input_data = input_pipelines[netD_id]

            # Enable gradient computation for the selected discriminator
            for p in netD.parameters():
                p.requires_grad = True

            # Train the discriminator
            for iter_d in range(config.critic_iters):
                real_data, real_y = next(input_data)
                batchsize = real_data.shape[0]
                real_data = real_data.view(batchsize, -1).to(self.device)
                real_y = real_y.to(self.device)
                real_data_v = autograd.Variable(real_data)

                dynamic_hook_function = dummy_hook
                netD.zero_grad()
                D_real_score = netD(real_data_v, real_y)
                D_real = -D_real_score.mean()

                # Generate fake data
                if self.latent_type == 'normal':
                    noise = torch.randn(batchsize, self.z_dim).to(self.device)
                elif self.latent_type == 'bernoulli':
                    noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
                else:
                    raise NotImplementedError
                noisev = autograd.Variable(noise)
                fake = autograd.Variable(netG(noisev, real_y.to(self.device)).view(batchsize, -1).data)
                inputv = fake.to(self.device)
                D_fake = netD(inputv, real_y.to(self.device))
                D_fake = D_fake.mean()

                # Compute the gradient penalty
                gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, config.L_gp, self.device)
                D_cost = D_fake + D_real + gradient_penalty

                # Add logit cost
                logit_cost = config.L_epsilon * torch.pow(D_real_score, 2).mean()
                D_cost += logit_cost

                # Backpropagate and update the discriminator
                D_cost.backward()
                Wasserstein_D = -D_real - D_fake
                optimizerD.step()

            # Clean up memory
            del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
            torch.cuda.empty_cache()

            # Switch to differential privacy hook
            dynamic_hook_function = dp_conv_hook

            # Disable gradient computation for the discriminator
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            # Train the generator
            if self.latent_type == 'normal':
                noise = torch.randn(batchsize, self.z_dim).to(self.device)
            elif self.latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
            else:
                raise NotImplementedError
            label = torch.randint(0, self.private_num_classes, [batchsize]).to(self.device)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, label).view(batchsize, -1)
            fake = fake.to(self.device)
            label = label.to(self.device)
            G = netD(fake, label)
            G = - G.mean()

            # Backpropagate and update the generator
            G.backward()
            G_cost = G
            optimizerG.step()

            # Apply exponential moving average to the generator
            exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

            # Log training progress
            if iter < 5 or iter % config.print_step == 0:
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))
                logging.info('Step: {}, G_cost:{}, D_cost:{}, Wasserstein:{}'.format(iter, G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))

            # Visualize generated images
            if iter % config.vis_step == 0:
                generate_image(iter, netGS, fix_noise, os.path.join(config.log_dir, "samples"), self.device, c=self.c, img_size=self.img_size, num_classes=self.private_num_classes)

            # Save model checkpoints
            if iter % config.save_step == 0:
                torch.save(netGS.state_dict(), os.path.join(config.log_dir, "checkpoints", 'netGS_%d.pth' % iter))

            # Clean up memory
            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        # Save final models
        torch.save(self.netG.state_dict(), os.path.join(config.log_dir, "checkpoints", 'netG.pth'))
        torch.save(self.netGS.state_dict(), os.path.join(config.log_dir, "checkpoints", 'netGS.pth'))


    def generate(self, config):
        # Create a directory to store logs and generated data
        os.mkdir(config.log_dir)
        
        # Initialize empty lists to hold synthetic data and labels
        syn_data = []
        syn_labels = []

        # Generate synthetic data and labels using the generator network (netGS)
        # Save the generated data to an NPZ file in the log directory
        syn_data, syn_labels = save_gen_data(
            os.path.join(config.log_dir, 'gen_data.npz'), 
            self.netGS, 
            self.z_dim, 
            self.device, 
            latent_type=self.latent_type, 
            c=self.c, 
            img_size=self.img_size, 
            num_classes=self.private_num_classes, 
            num_samples_per_class=config.data_num // self.private_num_classes
        )

        # Save the synthetic data and labels to another NPZ file for easy access
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        # Prepare a list of images to display one example from each class
        show_images = []
        for cls in range(self.private_num_classes):
            # Append the first 8 images of each class to the show_images list
            show_images.append(syn_data[syn_labels == cls][:8])
        # Concatenate all selected images into a single array
        show_images = np.concatenate(show_images)

        # Save the concatenated images as a grid image
        torchvision.utils.save_image(
            torch.from_numpy(show_images), 
            os.path.join(config.log_dir, 'sample.png'), 
            padding=1, 
            nrow=8
        )
        
        # Log a message indicating that the generation process is complete
        logging.info("Generation Finished!")

        # Return the synthetic data and labels
        return syn_data, syn_labels


def modify_gradnorm_conv_hook(module, grad_input, grad_output, CLIP_BOUND=1.0):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    # get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    # clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)

def master_hook_adder(module, grad_input, grad_output):
    '''
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    pass



def dp_conv_hook(module, grad_input, grad_output, CLIP_BOUND=1.0, SENSITIVITY=2.0):
    '''
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global noise_multiplier
    # get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    # clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    # add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)