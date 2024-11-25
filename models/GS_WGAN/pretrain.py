import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from data.stylegan3.dataset import ImageFolderDataset
from torch.utils.data import random_split

from models.GS_WGAN.models_ import *
from models.GS_WGAN.utils import *
from models.GS_WGAN.ops import exp_mov_avg
from models.DP_GAN.generator import Generator
from models.DP_GAN.discriminator import Discriminator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', '-s', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', '-data', type=str, default='mnist',
                        help=' dataset name')
    parser.add_argument('--num_discriminators', '-ndis', type=int, default=1000, help='number of discriminators')
    parser.add_argument('--noise_multiplier', '-noise', type=float, default=0., help='noise multiplier')
    parser.add_argument('--z_dim', '-zdim', type=int, default=60, help='latent code dimensionality')
    parser.add_argument('--c', type=int, default=1, help='latent code dimensionality')
    parser.add_argument('--img_size', type=int, default=28, help='latent code dimensionality')
    parser.add_argument('--private_num_classes', type=int, default=10, help='latent code dimensionality')
    parser.add_argument('--public_num_classes', type=int, default=10, help='latent code dimensionality')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--gpu_id', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--train_num', type=str, default="all", help='latent code dimensionality')
    parser.add_argument('--model_dim', '-mdim', type=int, default=64, help='model dimensionality')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='batch size')
    parser.add_argument('--L_gp', '-lgp', type=float, default=10, help='gradient penalty lambda hyperparameter')
    parser.add_argument('--L_epsilon', '-lep', type=float, default=0.001, help='epsilon penalty (used in PGGAN)')
    parser.add_argument('--critic_iters', '-diters', type=int, default=5, help='number of critic iters per gen iter')
    parser.add_argument('--latent_type', '-latent', type=str, default='bernoulli', choices=['normal', 'bernoulli'],
                        help='latent distribution')
    parser.add_argument('--iterations', '-iters', type=int, default=20000, help='iterations for training')
    parser.add_argument('--pretrain_iterations', '-piters', type=int, default=2000, help='iterations for pre-training')
    parser.add_argument('--num_workers', '-nwork', type=int, default=0, help='number of workers')
    parser.add_argument('--net_ids', '-ids', type=int, nargs='+', help='the index list for the discriminator')
    parser.add_argument('--print_step', '-pstep', type=int, default=100, help='number of steps to print')
    parser.add_argument('--vis_step', '-vstep', type=int, default=1000, help='number of steps to vis & eval')
    parser.add_argument('--save_step', '-sstep', type=int, default=5000, help='number of steps to save')
    parser.add_argument('--load_dir', '-ldir', type=str, help='checkpoint dir (for loading pre-trained models)')
    parser.add_argument('--pretrain', action='store_true', default=False, help='if performing pre-training')
    parser.add_argument('--num_gpus', '-ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gen_arch', '-gen', type=str, default='BigGAN', choices=['DCGAN', 'ResNet', 'BigGAN'],
                        help='generator architecture')
    parser.add_argument('--run', '-run', type=int, default=1, help='index number of run')
    parser.add_argument('--exp_name', '-name', type=str,
                        help='output folder name; will be automatically generated if not specified')
    args = parser.parse_args()
    return args

##########################################################
### main
##########################################################
def main(args):
    dataset = args.dataset
    num_discriminators = args.num_discriminators
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    save_dir = args.log_dir
    net_ids = args.net_ids
    gen_arch = args.gen_arch
    c = args.c
    img_size = args.img_size
    private_num_classes = args.private_num_classes
    public_num_classes = args.public_num_classes
    label_dim = max(private_num_classes, public_num_classes)
    train_num = args.train_num
    data_path = args.data_path

    ### Data loaders
    transform_train = transforms.ToTensor()
    trainset = ImageFolderDataset(args.data_path, img_size, c, use_labels=True)
    
    if train_num != "all":
        if "mnist" in dataset:
            train_size = 55000
        elif "cifar" in dataset:
            train_size = 45000
        elif "eurosat" in dataset:
            train_size = 21000
        elif "celeba" in dataset:
            train_size = 145064
        elif "camelyon" in dataset:
            train_size = 269538
        else:
            raise NotImplementedError

        val_size = len(trainset) - train_size
        torch.manual_seed(0)
        trainset, _ = random_split(trainset, [train_size, val_size])

    ### CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ### Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Fix noise for visualization
    if latent_type == 'normal':
        fix_noise = torch.randn(10, z_dim)
    elif latent_type == 'bernoulli':
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)
    else:
        raise NotImplementedError

    ### Set up models
    netD_list = []
    for i in range(len(net_ids)):
        netD = DiscriminatorDCGAN(c=c, img_size=img_size, num_classes=private_num_classes)
        netD_list.append(netD)
    netD_list = [netD.to(device) for netD in netD_list]

    ### Set up optimizers
    optimizerD_list = []
    for i in range(len(net_ids)):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)

    if os.path.exists(os.path.join(save_dir, 'indices.npy')):
        print('load indices from disk')
        indices_full = np.load(os.path.join(save_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('creat indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    ### Input pipelines
    input_pipelines = []
    for i in net_ids:
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Training Loop
    for idx, netD_id in enumerate(net_ids):

        ### stop the process if finished
        if netD_id >= num_discriminators:
            print('ID {} exceeds the num of discriminators'.format(netD_id))
            sys.exit()

        ### Discriminator
        netD = netD_list[idx]
        optimizerD = optimizerD_list[idx]
        input_data = input_pipelines[idx]

        ### Train (non-private) Generator for each Discriminator
        if gen_arch == 'DCGAN':
            netG = GeneratorDCGAN(c=c, img_size=img_size, z_dim=z_dim, model_dim=model_dim, num_classes=label_dim).to(device)
        elif gen_arch == 'ResNet':
            netG = GeneratorResNet(c=c, img_size=img_size, z_dim=z_dim, model_dim=model_dim, num_classes=label_dim).to(device)
        elif gen_arch == 'BigGAN':
            netG = Generator(z_dim=z_dim, img_size=img_size, num_classes=label_dim, g_conv_dim=model_dim, out=nn.Sigmoid())
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        ### Save dir for each discriminator
        save_subdir = os.path.join(save_dir, 'netD_%d' % netD_id)

        if os.path.exists(os.path.join(save_subdir, 'netD.pth')):
            print("netD %d already pre-trained" % netD_id)
        else:
            mkdir(save_subdir)

            for iter in range(args.pretrain_iterations + 1):
                #########################
                ### Update D network
                #########################
                for p in netD.parameters():
                    p.requires_grad = True

                for iter_d in range(critic_iters):
                    real_data, real_y = next(input_data)
                    batchsize = real_data.shape[0]
                    if len(real_y.shape) == 2:
                        real_data = real_data.to(torch.float32) / 255.
                        real_y = torch.argmax(real_y, dim=1)
                    # real_data = real_data * 2 - 1
                    real_data = real_data.view(batchsize, -1)
                    real_data = real_data.to(device)
                    real_y = real_y.to(device)
                    real_data_v = autograd.Variable(real_data)

                    ### train with real
                    netD.zero_grad()
                    D_real_score = netD(real_data_v, real_y)
                    D_real = -D_real_score.mean()

                    ### train with fake
                    batchsize = real_data.shape[0]
                    if latent_type == 'normal':
                        noise = torch.randn(batchsize, z_dim).to(device)
                    elif latent_type == 'bernoulli':
                        noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device)
                    else:
                        raise NotImplementedError
                    noisev = autograd.Variable(noise)
                    fake = netG(noisev, real_y)
                    fake = fake.view(batchsize, -1)
                    fake = autograd.Variable(fake.data)
                    inputv = fake
                    D_fake = netD(inputv, real_y)
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

                ############################
                # Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()

                if latent_type == 'normal':
                    noise = torch.randn(batchsize, z_dim).to(device)
                elif latent_type == 'bernoulli':
                    noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device)
                else:
                    raise NotImplementedError
                label = torch.randint(0, private_num_classes, [batchsize]).to(device)
                noisev = autograd.Variable(noise)
                fake = netG(noisev, label)
                fake = fake.view(batchsize, -1)
                G = netD(fake, label)
                G = - G.mean()

                ### update
                G.backward()
                G_cost = G
                optimizerG.step()

                ############################
                ### Results visualization
                ############################
                if iter < 5 or iter % args.print_step == 0:
                    print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                        D_cost.cpu().data,
                                                                        Wasserstein_D.cpu().data
                                                                        ))
                if iter == args.pretrain_iterations:
                    generate_image(iter, netG, fix_noise, save_subdir, device, c=c, img_size=img_size, num_classes=private_num_classes)

            torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)
