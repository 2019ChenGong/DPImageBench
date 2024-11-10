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
import concurrent.futures

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier

from models.GS_WGAN.models import *
from models.GS_WGAN.utils import *
from models.GS_WGAN.ops import exp_mov_avg
from models.DP_GAN.generator import Generator

from models.synthesizer import DPSynther

class GS_WGAN(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.num_discriminators = config.num_discriminators
        self.z_dim = config.Generator.z_dim
        self.c = config.c
        self.img_size =  config.img_size
        self.num_classes = config.num_classes
        self.latent_type = config.latent_type
        self.ckpt = config.ckpt

        self.netG = Generator(img_size=self.img_size, num_classes=self.num_classes, **config.Generator)
        
        self.netGS = copy.deepcopy(self.netG)
        self.netD_list = []
        for i in range(self.num_discriminators):
            netD = DiscriminatorDCGAN(c=self.c, img_size=self.img_size, num_classes=self.num_classes)
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
        netD = DiscriminatorDCGAN(c=self.c, img_size=self.img_size, num_classes=self.num_classes).to(self.device)
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
                if config.cond:
                    real_y = real_y % self.num_classes
                else:
                    real_y = torch.zeros_like(real_y).long()
                real_data = real_data * 2 - 1
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
                print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                    D_cost.cpu().data,
                                                                    Wasserstein_D.cpu().data
                                                                    ))
                logging.info('Step: {}, G_cost:{}, D_cost:{}, Wasserstein:{}'.format(iter, G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data))

            if iter % config.vis_step == 0:
                generate_image(iter, netGS, fix_noise, config.log_dir, self.device, c=self.c, img_size=self.img_size, num_classes=self.num_classes)

            del label, fake, noisev, noise, G, G_cost, D_cost
            torch.cuda.empty_cache()

        ### save model
        torch.save(self.netG.state_dict(), os.path.join(config.log_dir, 'netG.pth'))
        torch.save(self.netGS.state_dict(), os.path.join(config.log_dir, 'netGS.pth'))

    def warmup_one_discriminator(self):
        print(11111)
        ### Fix noise for visualization
        if self.latent_type == 'normal':
            fix_noise = torch.randn(10, self.z_dim)
        elif self.latent_type == 'bernoulli':
            p = 0.5
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((10, self.z_dim)).view(10, self.z_dim)
        else:
            raise NotImplementedError

        trainset_size = int(len(trainset) / self.num_discriminators)

        ### Training Loop
        for idx, netD_id in enumerate(net_ids):

            ### stop the process if finished
            if netD_id >= self.num_discriminators:
                logging.info('ID {} exceeds the num of discriminators'.format(netD_id))
                return

            ### Discriminator
            netD = self.netD_list[netD_id]
            optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

            start = netD_id * trainset_size
            end = (netD_id + 1) * trainset_size
            indices = indices_full[start:end]
            trainloader = data.DataLoader(trainset, batch_size=config.batchsize, drop_last=False,
                                            num_workers=2, sampler=SubsetRandomSampler(indices))
            input_data = inf_train_gen(trainloader)

            ### Train (non-private) Generator for each Discriminator
            netG = copy.deepcopy(self.netG)
            optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

            ### Save dir for each discriminator
            save_subdir = os.path.join(config.log_dir, 'netD_%d' % netD_id)

            if os.path.exists(os.path.join(save_subdir, 'netD.pth')):
                logging.info("netD %d already pre-trained" % netD_id)
            else:
                mkdir(save_subdir)

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

                        batchsize = real_data.shape[0]
                        real_data = real_data.view(batchsize, -1)
                        real_data = real_data.to(self.device)
                        real_y = real_y.to(self.device)
                        real_data_v = autograd.Variable(real_data)

                        ### train with real
                        netD.zero_grad()
                        D_real_score = netD(real_data_v, real_y)
                        D_real = -D_real_score.mean()

                        ### train with fake
                        batchsize = real_data.shape[0]
                        if latent_type == 'normal':
                            noise = torch.randn(batchsize, self.z_dim).to(self.device)
                        elif latent_type == 'bernoulli':
                            noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
                        else:
                            raise NotImplementedError
                        noisev = autograd.Variable(noise)
                        fake = autograd.Variable(netG(noisev, real_y).data)
                        inputv = fake
                        D_fake = netD(inputv, real_y)
                        D_fake = D_fake.mean()

                        ### train with gradient penalty
                        gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, self.device)
                        D_cost = D_fake + D_real + gradient_penalty

                        ### train with epsilon penalty
                        logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
                        D_cost += logit_cost

                        ### update
                        D_cost.backward()
                        Wasserstein_D = -D_real - D_fake
                        optimizerD.step()

                    ############################
                    # Update G network
                    ###########################
                    for p in netD.parameters():
                        p.requires_grad = False
                    netG.zero_grad()

                    if latent_type == 'normal':
                        noise = torch.randn(batchsize, self.z_dim).to(self.device)
                    elif latent_type == 'bernoulli':
                        noise = bernoulli.sample((batchsize, self.z_dim)).view(batchsize, self.z_dim).to(self.device)
                    else:
                        raise NotImplementedError
                    label = torch.randint(0, self.num_classes, [batchsize]).to(self.device)
                    noisev = autograd.Variable(noise)
                    fake = netG(noisev, label)
                    G = netD(fake, label)
                    G = - G.mean()

                    ### update
                    G.backward()
                    G_cost = G
                    optimizerG.step()

                    ############################
                    ### Results visualization
                    ############################
                    if iter < 5 or iter % config.print_step == 0:
                        logging.info('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                            D_cost.cpu().data,
                                                                            Wasserstein_D.cpu().data
                                                                            ))
                    if iter == config.iterations:
                        generate_image(iter, netGS, fix_noise, config.log_dir, self.device, c=self.c, img_size=self.img_size, num_classes=self.num_classes)

                torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))

    def warmup(self, sensitive_dataset, indices_full, trainset_size, config):

        ### Set up optimizers
        njobs = config.njobs
        dis_per_job = self.num_discriminators // njobs
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.warmup_one_discriminator, sensitive_dataset, indices_full, [j+i*dis_per_job for j in range(dis_per_job)], config) for i in range(njobs)]
            # futures = [executor.submit(ss) for i in range(njobs)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    score = future.result()
                    logging.info("jobs is finished")
                except Exception as e:
                    print(f'Error: {e}')

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        os.mkdir(config.log_dir)
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
                real_data = real_data * 2 - 1
                
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