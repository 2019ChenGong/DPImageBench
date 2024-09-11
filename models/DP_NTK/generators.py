import torch as pt
import torch.nn as nn


class ConvCondGen(nn.Module):
    def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
        super(ConvCondGen, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(self.nc) == 3 and len(self.ks) == 2
        self.hw = 7  # image height and width before upsampling
        self.reshape_size = self.nc[0] * self.hw ** 2
        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0] - 1) // 2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1] - 1) // 2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x
        # print(x.shape)
        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device='cuda', return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
        code = pt.randn(batch_size, self.d_code, device=device)
        gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code