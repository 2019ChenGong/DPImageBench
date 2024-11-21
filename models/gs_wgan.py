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
from concurrent.futures import ProcessPoolExecutor

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
        super().__init__()

        self.num_discriminators = config.num_discriminators
        self.z_dim = config.Generator.z_dim
        self.c = config.c
        self.img_size =  config.img_size
        self.private_num_classes = config.private_num_classes
        self.public_num_classes = config.public_num_classes
        self.label_dim = max(self.private_num_classes, self.public_num_classes)
        self.latent_type = config.latent_type
        self.ckpt = config.ckpt

        # self.netG = Generator(img_size=self.img_size, num_classes=label_dim, **config.Generator)
        self.netG = GeneratorResNet(c=self.c, img_size=self.img_size, z_dim=self.z_dim, model_dim=config.Generator.g_conv_dim, num_classes=self.label_dim)
        
        self.netGS = copy.deepcopy(self.netG)
        self.netD_list = []
        for i in range(self.num_discriminators):
            netD = DiscriminatorDCGAN(c=self.c, img_size=self.img_size, num_classes=self.label_dim)
            self.netD_list.append(netD)
        
        model_parameters = filter(lambda p: p.requires_grad, self.netG.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)

        self.config = config
        self.device = device
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)

        ### Random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        ### Fix noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError

        netG = self.netG.to(self.device)
        netGS = self.netGS.to(self.device)

        ### Set up optimizers
        netD = DiscriminatorDCGAN(c=self.c, img_size=self.img_size, num_classes=self.label_dim).to(self.device)
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        ### Register hook
        global dynamic_hook_function
        netD.conv1.register_backward_hook(master_hook_adder)

        input_data = inf_train_gen(public_dataloader)

        for iter in range(config.iterations + 1):
            #########################
            ### Update D network
            #########################

            for p in netD.parameters():
                p.requires_grad = True

            for iter_d in range(config.critic_iters):
                real_data, real_y = next(input_data)
                if len(real_y.shape) == 2:
                    real_data = real_data.to(torch.float32) / 255.
                    real_y = torch.argmax(real_y, dim=1)
                if not config.cond:
                    real_y = torch.zeros_like(real_y).long()
                # real_data = real_data * 2 - 1
                batchsize = real_data.shape[0]
                real_data = real_data.view(batchsize, -1)
                real_data = real_data.to(self.device)
                real_y = real_y.to(self.device)
                real_data_v = autograd.Variable(real_data)

                ### train with real
                dynamic_hook_function = dummy_hook
                netD.zero_grad()
                D_real_score = netD(real_data_v, real_y)
                D_real = -D_real_score.mean()

                ### train with fake
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

                ### train with gradient penalty
                gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, config.L_gp, self.device)
                D_cost = D_fake + D_real + gradient_penalty

                ### train with epsilon penalty
                logit_cost = config.L_epsilon * torch.pow(D_real_score, 2).mean()
                D_cost += logit_cost

                ### update
                D_cost.backward()
                Wasserstein_D = -D_real - D_fake
                optimizerD.step()

            del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
            torch.cuda.empty_cache()

            ############################
            # Update G network
            ###########################
            dynamic_hook_function = modify_gradnorm_conv_hook

            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            ### train with sanitized discriminator output
            if self.latent_type == 'normal':
                noise = torch.randn(batchsize, self.z_dim).to(self.device)
            elif self.latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
            else:
                raise NotImplementedError
            label = torch.randint(0, self.public_num_classes, [batchsize]).to(self.device)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, label).view(batchsize, -1)
            fake = fake.to(self.device)
            label = label.to(self.device)
            G = netD(fake, label)
            G = - G.mean()

            ### update
            G.backward()
            G_cost = G
            optimizerG.step()

            ### update the exponential moving average
            exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

            ############################
            ### Results visualization
            ############################
            if iter < 5 or iter % config.print_step == 0:
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                    D_cost.cpu().data,
                                                                    Wasserstein_D.cpu().data
                                                                    ))
                logging.info('Step: {}, G_cost:{}, D_cost:{}, Wasserstein:{}'.format(iter, G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))

            if iter % config.vis_step == 0:
                generate_image(iter, netGS, fix_noise, config.log_dir, self.device, c=self.c, img_size=self.img_size, num_classes=self.public_num_classes)

            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        ### save model
        torch.save(self.netG.state_dict(), os.path.join(config.log_dir, 'netG.pth'))
        torch.save(self.netGS.state_dict(), os.path.join(config.log_dir, 'netGS.pth'))
    
    def warmup_training(self, config):
        n_gpu = config.n_gpu
        iters = str(config.iterations)
        data_name = config.data_name
        train_num = config.eval_mode
        data_path = config.data_path
        gen_arch = "BigGAN"
        img_size = str(self.img_size)
        c = str(self.c)
        num_classes = str(self.num_classes)
        log_dir = config.log_dir
        ndis = str(self.num_discriminators)
        dis_per_job=200 # number of discriminators to be trained for each process
        njobs = self.num_discriminators // dis_per_job
        scripts = []
        for gpu_id in range(n_gpu):
            meta_start = dis_per_job // n_gpu * gpu_id
            for job_id in range(njobs):
                start= (job_id * dis_per_job + meta_start)
                end= (start + dis_per_job)
                vals= [str(dis_id) for dis_id in range(start, end)]
                script = ['models/GS_WGAN/pretrain.py', '-data', data_name, '--log_dir', log_dir, '--train_num', train_num, '-ndis', ndis, '-ids'] + vals + ['--img_size', img_size, '--c', c, '--num_classes', num_classes, '--gpu_id', str(gpu_id), '--data_path', data_path, '-piters', iters, '--gen_arch', gen_arch, '--z_dim', str(self.z_dim), '--latent_type', self.latent_type, '--model_dim', str(self.config.Generator.g_conv_dim)]
                scripts.append(script)
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(warm_up, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    print(f"Output:\n{output}")
                except Exception as e:
                    print(f"generated an exception: {e}")



    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        os.mkdir(config.log_dir)

        if self.ckpt is None:
            config.pretrain.log_dir = os.path.join(config.log_dir, 'warm_up')
            os.mkdir(config.pretrain.log_dir)
            indices_full = np.arange(len(sensitive_dataloader.dataset))
            np.random.shuffle(indices_full)
            indices_full.dump(os.path.join(config.pretrain.log_dir, 'indices.npy'))
            self.warmup_training(config.pretrain)
            load_dir = config.pretrain.log_dir
        else:
            load_dir = self.ckpt
            indices_full = np.load(os.path.join(load_dir, 'indices.npy'), allow_pickle=True)
        if len(indices_full) != len(sensitive_dataloader.dataset):
            indices_full = np.arange(len(sensitive_dataloader.dataset))
            np.random.shuffle(indices_full)
            indices_full.dump(os.path.join(config.log_dir, 'indices.npy'))
        trainset_size = int(len(sensitive_dataloader.dataset) / self.num_discriminators)
        logging.info('Size of the dataset: {}'.format(trainset_size))

        # self.warmup(sensitive_dataloader.dataset, indices_full, trainset_size, config.pretrain)

        self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1./self.num_discriminators, steps=config.iterations)
        global noise_multiplier
        noise_multiplier = self.noise_factor
        logging.info("The noise factor is {}".format(self.noise_factor))

        ### Random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        ### Fix noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError

        netG = self.netG.to(self.device)
        netGS = self.netGS.to(self.device)
        for netD_id, netD in enumerate(self.netD_list):
            self.netD_list[netD_id] = netD.to(self.device)

        ### Set up optimizers
        optimizerD_list = []
        for i in range(self.num_discriminators):
            netD = self.netD_list[i]
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD.load_state_dict(torch.load(network_path))
            optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
            optimizerD_list.append(optimizerD)
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        input_pipelines = []
        dataset = sensitive_dataloader.dataset
        for i in range(self.num_discriminators):
            start = i * trainset_size
            end = (i + 1) * trainset_size
            indices = indices_full[start:end]
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, drop_last=False, sampler=SubsetRandomSampler(indices))
            input_data = inf_train_gen(trainloader)
            input_pipelines.append(input_data)

        ### Register hook
        global dynamic_hook_function
        for netD in self.netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)

        for iter in range(config.iterations + 1):
            #########################
            ### Update D network
            #########################
            netD_id = np.random.randint(self.num_discriminators, size=1)[0]
            netD = self.netD_list[netD_id]
            optimizerD = optimizerD_list[netD_id]
            input_data = input_pipelines[netD_id]

            for p in netD.parameters():
                p.requires_grad = True

            for iter_d in range(config.critic_iters):
                real_data, real_y = next(input_data)
                if len(real_y.shape) == 2:
                    real_data = real_data.to(torch.float32) / 255.
                    real_y = torch.argmax(real_y, dim=1)
                # real_data = real_data * 2 - 1
                
                batchsize = real_data.shape[0]
                real_data = real_data.view(batchsize, -1)
                real_data = real_data.to(self.device)
                real_y = real_y.to(self.device)
                real_data_v = autograd.Variable(real_data)

                ### train with real
                dynamic_hook_function = dummy_hook
                netD.zero_grad()
                D_real_score = netD(real_data_v, real_y)
                D_real = -D_real_score.mean()

                ### train with fake
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

                ### train with gradient penalty
                gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, config.L_gp, self.device)
                D_cost = D_fake + D_real + gradient_penalty

                ### train with epsilon penalty
                logit_cost = config.L_epsilon * torch.pow(D_real_score, 2).mean()
                D_cost += logit_cost

                ### update
                D_cost.backward()
                Wasserstein_D = -D_real - D_fake
                optimizerD.step()

            del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
            torch.cuda.empty_cache()

            ############################
            # Update G network
            ###########################
            dynamic_hook_function = dp_conv_hook

            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            ### train with sanitized discriminator output
            if self.latent_type == 'normal':
                noise = torch.randn(batchsize, self.z_dim).to(self.device)
            elif self.latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
            else:
                raise NotImplementedError
            label = torch.randint(0, self.num_classes, [batchsize]).to(self.device)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, label).view(batchsize, -1)
            fake = fake.to(self.device)
            label = label.to(self.device)
            G = netD(fake, label)
            G = - G.mean()

            ### update
            G.backward()
            G_cost = G
            optimizerG.step()

            ### update the exponential moving average
            exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

            ############################
            ### Results visualization
            ############################
            if iter < 5 or iter % config.print_step == 0:
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))
                logging.info('Step: {}, G_cost:{}, D_cost:{}, Wasserstein:{}'.format(iter, G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))

            if iter % config.vis_step == 0:
                generate_image(iter, netGS, fix_noise, config.log_dir, self.device, c=self.c, img_size=self.img_size, num_classes=self.num_classes)

            if iter % config.save_step == 0:
                ### save model
                torch.save(netGS.state_dict(), os.path.join(config.log_dir, 'netGS_%d.pth' % iter))

            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        ### save model
        torch.save(self.netG.state_dict(), os.path.join(config.log_dir, 'netG.pth'))
        torch.save(self.netGS.state_dict(), os.path.join(config.log_dir, 'netGS.pth'))

    def generate(self, config):
        os.mkdir(config.log_dir)
        syn_data = []
        syn_labels = []

        syn_data, syn_labels = save_gen_data(os.path.join(config.log_dir, 'gen_data.npz'), self.netGS, self.z_dim, self.device, latent_type=self.latent_type, c=self.c, img_size=self.img_size, num_classes=self.num_classes, num_samples_per_class=config.data_num//self.num_classes)

        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.num_classes):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        logging.info("Generation Finished!")
        return syn_data, syn_labels


##########################################################
### hook functions
##########################################################

def modify_gradnorm_conv_hook(module, grad_input, grad_output, CLIP_BOUND=1.0):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
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
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    ### add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)