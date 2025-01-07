# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/big_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from models.DP_GAN import ops
from models.DP_GAN import misc


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES):
        super(DiscOptBlock, self).__init__()
        self.apply_d_sn = apply_d_sn

        self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            #self.bn0 = MODULES.d_bn(in_features=in_channels)
            #self.bn1 = MODULES.d_bn(in_features=out_channels)
            self.bn0 = nn.GroupNorm(min(32, in_channels), in_channels)
            self.bn1 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.activation = MODULES.d_act_fn
        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if not self.apply_d_sn:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES, downsample=True):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn
        self.downsample = downsample

        self.activation = MODULES.d_act_fn

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if self.ch_mismatch or downsample:
            self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if not apply_d_sn:
                # self.bn0 = MODULES.d_bn(in_features=in_channels)
                self.bn0 = nn.GroupNorm(min(32, in_channels), in_channels)

        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            #self.bn1 = MODULES.d_bn(in_features=in_channels)
            #self.bn2 = MODULES.d_bn(in_features=out_channels)
            self.bn1 = nn.GroupNorm(min(32, in_channels), in_channels)
            self.bn2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)

        if not self.apply_d_sn:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if not self.apply_d_sn:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)
        out = x + x0
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size=32, d_conv_dim=96, apply_d_sn=False, apply_attn=False, attn_d_loc=[1], d_embed_dim=None, normalize_d_embed=False,
                 num_classes=10, d_init="ortho"):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {
            "28": [1] + [d_conv_dim * 2, d_conv_dim * 2],
            "32": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "128": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            # "64": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            # "128": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
            "512": [3] + [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
        }

        d_out_dims_collection = {
            "28": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "32": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "128": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            # "64": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            # "128": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "512":
            [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
        }

        d_down = {
            "28": [True, True, False],
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        self.MODULES = misc.make_empty_object()
        self.apply_d_sn = apply_d_sn
        self.d_cond_mtd = "PD"
        self.define_modules()
        MODULES = self.MODULES

        self.normalize_d_embed = normalize_d_embed
        self.num_classes = num_classes
        self.in_dims = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[
                    DiscOptBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], apply_d_sn=apply_d_sn, MODULES=MODULES)
                ]]
            else:
                self.blocks += [[
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              apply_d_sn=apply_d_sn,
                              MODULES=MODULES,
                              downsample=down[index])
                ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn

        # linear layer for adversarial training
        self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=1, bias=True)

        # linear and embedding layers for discriminator conditioning
        if self.d_cond_mtd == "AC":
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
        elif self.d_cond_mtd == "PD":
            self.embedding = MODULES.d_embedding(num_classes, self.out_dims[-1])
        elif self.d_cond_mtd in ["2C", "D2DCE"]:
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=d_embed_dim, bias=True)
            self.embedding = MODULES.d_embedding(num_classes, d_embed_dim)

        if d_init:
            ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        with misc.dummy_context_mgr() as mp:
            embed, proxy, cls_output = None, None, None
            mi_embed, mi_proxy, mi_cls_output = None, None, None
            info_discrete_c_logits, info_conti_mu, info_conti_var = None, None, None
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)

            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            # adversarial training
            adv_output = torch.squeeze(self.linear1(h))

            # class conditioning
            if self.d_cond_mtd == "AC":
                if self.normalize_d_embed:
                    for W in self.linear2.parameters():
                        W = F.normalize(W, dim=1)
                    h = F.normalize(h, dim=1)
                cls_output = self.linear2(h)
            elif self.d_cond_mtd == "PD":
                adv_output = adv_output + torch.sum(torch.mul(self.embedding(label), h), 1)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                embed = self.linear2(h)
                proxy = self.embedding(label)
                if self.normalize_d_embed:
                    embed = F.normalize(embed, dim=1)
                    proxy = F.normalize(proxy, dim=1)
            elif self.d_cond_mtd == "MD":
                idx = torch.LongTensor(range(label.size(0))).to(label.device)
                adv_output = adv_output[idx, label]
            elif self.d_cond_mtd in ["W/O", "MH"]:
                pass
            else:
                raise NotImplementedError

        return adv_output

    def define_modules(self):

        if self.apply_d_sn:
            self.MODULES.d_conv2d = ops.snconv2d
            self.MODULES.d_deconv2d = ops.sndeconv2d
            self.MODULES.d_linear = ops.snlinear
            self.MODULES.d_embedding = ops.sn_embedding
        else:
            self.MODULES.d_conv2d = ops.conv2d
            self.MODULES.d_deconv2d = ops.deconv2d
            self.MODULES.d_linear = ops.linear
            self.MODULES.d_embedding = ops.embedding

        if not self.apply_d_sn:
            self.MODULES.d_bn = ops.batchnorm_2d

        self.MODULES.d_act_fn = nn.ReLU(inplace=True)
        return self.MODULES
    
    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1, 1, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty


if __name__ == "__main__":
    import numpy as np
    d = Discriminator(img_size=28, num_classes=10, )
    model_parameters = filter(lambda p: p.requires_grad, d.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters in D: {}".format(n_params))