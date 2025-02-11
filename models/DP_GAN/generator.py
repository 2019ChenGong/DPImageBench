# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/big_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DP_GAN import ops
from models.DP_GAN import misc


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, affine_input_dim, MODULES):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = g_cond_mtd

        self.bn1 = MODULES.g_bn(affine_input_dim, in_channels, MODULES)
        self.bn2 = MODULES.g_bn(affine_input_dim, out_channels, MODULES)

        self.activation = MODULES.g_act_fn
        self.conv2d0 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.g_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=80, g_shared_dim=128, img_size=32, g_conv_dim=96, apply_attn=True, attn_g_loc=[2], num_classes=10, g_init="ortho", out=nn.Tanh()):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "28": [g_conv_dim * 4, g_conv_dim * 4],
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "128": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            # "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            # "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "28": [g_conv_dim * 4, g_conv_dim * 4],
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "128": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            # "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            # "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"28": 7, "32": 4, "64": 8, "128": 16, "256": 4, "512": 4}

        self.MODULES = misc.make_empty_object()
        self.apply_g_sn = True
        self.g_cond_mtd = "cBN"
        self.define_modules()
        MODULES = self.MODULES

        self.z_dim = z_dim
        self.g_shared_dim = g_shared_dim
        self.num_classes = num_classes
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.chunk_size = z_dim // (self.num_blocks + 1)
        self.affine_input_dim = self.chunk_size
        assert self.z_dim % (self.num_blocks + 1) == 0, "z_dim should be divided by the number of blocks"

        self.linear0 = MODULES.g_linear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom, bias=True)

        if self.g_cond_mtd != "W/O":
            self.affine_input_dim += self.g_shared_dim
            self.shared = ops.embedding(num_embeddings=self.num_classes, embedding_dim=self.g_shared_dim)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         g_cond_mtd=self.g_cond_mtd,
                         affine_input_dim=self.affine_input_dim,
                         MODULES=MODULES)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = MODULES.g_act_fn
        self.conv2d5 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=3 if img_size != 28 else 1, kernel_size=3, stride=1, padding=1)
        self.tanh = out

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []
        with misc.dummy_context_mgr() as mp:
            zs = torch.split(z, self.chunk_size, 1)
            z = zs[0]
            if self.g_cond_mtd != "W/O":
                if shared_label is None:
                    shared_label = self.shared(label)
                affine_list.append(shared_label)
            if len(affine_list) == 0:
                affines = [item for item in zs[1:]]
            else:
                affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affines[counter])
                        counter += 1

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out

    def define_modules(self):
        if self.apply_g_sn:
            self.MODULES.g_conv2d = ops.snconv2d
            self.MODULES.g_deconv2d = ops.sndeconv2d
            self.MODULES.g_linear = ops.snlinear
            self.MODULES.g_embedding = ops.sn_embedding
        else:
            self.MODULES.g_conv2d = ops.conv2d
            self.MODULES.g_deconv2d = ops.deconv2d
            self.MODULES.g_linear = ops.linear
            self.MODULES.g_embedding = ops.embedding

        if self.g_cond_mtd == "cBN":
            self.MODULES.g_bn = ops.ConditionalBatchNorm2d
        elif self.g_cond_mtd == "W/O":
            self.MODULES.g_bn = ops.batchnorm_2d
        elif self.g_cond_mtd == "cAdaIN":
            pass
        else:
            raise NotImplementedError

        self.MODULES.g_act_fn = nn.ReLU(inplace=True)
        return self.MODULES


if __name__ == "__main__":
    import numpy as np
    g = Generator(img_size=28, num_classes=10, z_dim=60, g_conv_dim=50)
    model_parameters = filter(lambda p: p.requires_grad, g.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters in G: {}".format(n_params))