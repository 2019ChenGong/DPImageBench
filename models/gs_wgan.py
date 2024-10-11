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

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier

from models.GS_WGAN.models import *
from models.GS_WGAN.utils import *
from models.GS_WGAN.ops import exp_mov_avg

from models.synthesizer import DPSynther

class DP_Kernel(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.num_discriminators = num_discriminators
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.latent_type = latent_type
        self.gen_arch = gen_arch
        self.ckpt = ckpt

        if self.gen_arch == 'DCGAN':
            self.netG = GeneratorDCGAN(z_dim=self.z_dim, model_dim=self.model_dim, num_classes=self.num_classes)
        elif self.gen_arch == 'ResNet':
            self.netG = GeneratorResNet(z_dim=self.z_dim, model_dim=self.model_dim, num_classes=self.num_classes)
        else:
            raise ValueError
        
        self.netGS = copy.deepcopy(netG)
        self.netG = self.netG
        self.netGS = self.netGS
        self.netD_list = []
        for i in range(self.num_discriminators):
            netD = DiscriminatorDCGAN()
            self.netD_list.append(netD)
        
        if ckpt is not None:
            for netD_id in range(self.num_discriminators):
                logging.info('Load NetD ', str(netD_id))
                network_path = os.path.join(ckpt, 'netD_%d' % netD_id, 'netD.pth')
                netD = self.netD_list[netD_id]
                netD.load_state_dict(torch.load(network_path))
            assert os.path.exists(os.path.join(ckpt, 'indices.npy'))
            self.indices_full = np.load(os.path.join(ckpt, 'indices.npy'), allow_pickle=True)
        else:
            self.indices_full = np.arange(len(trainset))
            np.random.shuffle(self.indices_full)

        self.config = config
        self.device = device
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)

        fixed_noise = torch.randn(8 * self.n_class, self.nz).to(self.device)
        fixed_label = torch.arange(self.n_class).repeat(8).to(self.device)

        optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=config.lr)
        for epoch in range(config.n_epochs):
            for x, label in public_dataloader:
                iter_loss = 0

                if len(label.shape) == 2:
                    x = x.to(torch.float32) / 255.
                    label = torch.argmax(label, dim=1)
                if x.shape[1] == 1:
                    x = F.interpolate(x, size=[32, 32])
                if config.label_random:
                    label = label % self.n_class
                x = x.to(self.device) * 2 - 1
                label = label.to(self.device)
                batch_size = x.size(0)

                gen_labels = get_gen_labels(label, self.n_class)

                optimizer.zero_grad()

                noise = torch.randn(batch_size, self.nz).to(self.device)
                y = self.gen(noise, label=gen_labels)
                label = F.one_hot(label, self.n_class).float()
                gen_labels = F.one_hot(gen_labels, self.n_class).float()

                #### compute mmd loss using my implementation ####
                DP_mmd_loss = rbf_kernel_DP_loss_with_labels(x.view(batch_size, -1), y.view(batch_size, -1), label, gen_labels, self.sigma_list, 0.)

                errG = torch.pow(DP_mmd_loss, 2)
                errG.backward()
                optimizer.step()
                iter_loss += errG.item()

            logging.info('Training loss: {}\tLoss:{:.6f}\t'.format(epoch, iter_loss))
            y_fixed = self.gen(fixed_noise, label=fixed_label)
            y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            grid = torchvision.utils.make_grid(y_fixed.data, nrow=10)
            torchvision.utils.save_image(grid, os.path.join(config.log_dir, f'netG_epoch{epoch}_lr{config.lr}_bs{config.batch_size}.png'))
            torch.save(self.gen.state_dict(), os.path.join(config.log_dir, f'netG_epoch{epoch}_lr{config.lr}_bs{config.batch_size}.pkl'))


    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        os.mkdir(config.log_dir)

        ### CUDA
        use_cuda = torch.cuda.is_available()
        devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
        device0 = devices[0]
        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ### Random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        ### Fix noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError

        netG = self.netG.to(device0)
        for netD_id, netD in enumerate(self.netD_list):
            device = devices[get_device_id(netD_id, self.num_discriminators, num_gpus)]
            netD.to(device)

        ### Set up optimizers
        optimizerD_list = []
        for i in range(self.num_discriminators):
            netD = self.netD_list[i]
            optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
            optimizerD_list.append(optimizerD)
        optimizerG = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if self.ckpt is not None:
            assert os.path.exists(os.path.join(self.ckpt, 'indices.npy'))
            indices_full = np.load(os.path.join(self.ckpt, 'indices.npy'), allow_pickle=True)
        else:
            indices_full = np.arange(len(sensitive_dataloader.dataset))
            np.random.shuffle(indices_full)
            indices_full.dump(os.path.join(config.log_dir, 'indices.npy'))
        trainset_size = int(len(sensitive_dataloader.dataset) / self.num_discriminators)
        logging.info('Size of the dataset: ', trainset_size)

        input_pipelines = []
        for i in range(self.num_discriminators):
            start = i * trainset_size
            end = (i + 1) * trainset_size
            indices = indices_full[start:end]
            trainloader = torch.utils.data.DataLoader(sensitive_dataloader.trainset, batch_size=config.batchsize, drop_last=False, sampler=SubsetRandomSampler(indices))
            input_data = inf_train_gen(trainloader)
            input_pipelines.append(input_data)

        ### Register hook
        global dynamic_hook_function
        for netD in self.netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)

        for iter in range(iterations + 1):
            #########################
            ### Update D network
            #########################
            netD_id = np.random.randint(self.num_discriminators, size=1)[0]
            device = devices[get_device_id(netD_id, self.num_discriminators, num_gpus)]
            netD = self.netD_list[netD_id]
            optimizerD = optimizerD_list[netD_id]
            input_data = input_pipelines[netD_id]

            for p in netD.parameters():
                p.requires_grad = True

            for iter_d in range(critic_iters):
                real_data, real_y = next(input_data)
                if len(real_y.shape) == 2:
                    real_data = real_data.to(torch.float32) / 255.
                    real_y = torch.argmax(real_y, dim=1)
                
                batchsize = real_data.shape[0]
                real_data = real_data.view(batchsize, -1)
                real_data = real_data.to(device)
                real_y = real_y.to(device)
                real_data_v = autograd.Variable(real_data)

                ### train with real
                dynamic_hook_function = dummy_hook
                netD.zero_grad()
                D_real_score = netD(real_data_v, real_y)
                D_real = -D_real_score.mean()

                ### train with fake
                if self.latent_type == 'normal':
                    noise = torch.randn(batchsize, self.z_dim).to(device0)
                elif self.latent_type == 'bernoulli':
                    noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(sdevice0)
                else:
                    raise NotImplementedError
                noisev = autograd.Variable(noise)
                fake = autograd.Variable(netG(noisev, real_y.to(device0)).data)
                inputv = fake.to(device)
                D_fake = netD(inputv, real_y.to(device))
                D_fake = D_fake.mean()

                ### train with gradient penalty
                gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, device)
                D_cost = D_fake + D_real + gradient_penalty

                ### train with epsilon penalty
                logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
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
            if if_dp:
                ### Sanitize the gradients passed to the Generator
                dynamic_hook_function = dp_conv_hook
            else:
                ### Only modify the gradient norm, without adding noise
                dynamic_hook_function = modify_gradnorm_conv_hook

            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            ### train with sanitized discriminator output
            if self.latent_type == 'normal':
                noise = torch.randn(batchsize, self.z_dim).to(device0)
            elif self.latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(device0)
            else:
                raise NotImplementedError
            label = torch.randint(0, self.num_classes, [batchsize]).to(device0)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, label)
            fake = fake.to(device)
            label = label.to(device)
            G = netD(fake, label)
            G = - G.mean()

            ### update
            G.backward()
            G_cost = G
            optimizerG.step()

            ### update the exponential moving average
            exp_mov_avg(self.netGS, netG, alpha=0.999, global_step=iter)

            ############################
            ### Results visualization
            ############################
            if iter < 5 or iter % args.print_step == 0:
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                    D_cost.cpu().data,
                                                                    Wasserstein_D.cpu().data
                                                                    ))

            if iter % args.vis_step == 0:
                generate_image(iter, netGS, fix_noise, save_dir, device0, num_classes=10)

            if iter % args.save_step == 0:
                ### save model
                torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS_%d.pth' % iter))

            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        ### save model
        torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
        torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS.pth'))

        ### save generate samples
        save_gen_data(os.path.join(save_dir, 'gen_data.npz'), netGS, z_dim, device0, latent_type=latent_type)




        def generate(self, config):
            os.mkdir(config.log_dir)
            syn_data = []
            syn_labels = []

            for _ in range(int(config.data_num / config.batch_size)):
                y = torch.randint(self.n_class, (config.batch_size,)).to(self.device)
                z = torch.randn(config.batch_size, self.nz).to(self.device)
                images = self.gen(z, y)
                if images.shape[1] == 1:
                    images = F.interpolate(images, size=[28, 28])

                syn_data.append(images.detach().cpu().numpy())
                syn_labels.append(y.detach().cpu().numpy())
            
            syn_data = np.concatenate(syn_data) / 2 + 0.5
            syn_labels = np.concatenate(syn_labels)

            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

            show_images = []
            for cls in range(self.n_class):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            return syn_data, syn_labels


##########################################################
### hook functions
##########################################################
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


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
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


def dp_conv_hook(module, grad_input, grad_output):
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