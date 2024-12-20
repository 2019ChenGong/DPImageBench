from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random
from tqdm import tqdm


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def set_seeds(rank, seed):
    random.seed(rank + seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


class MLP(nn.Module):

    def __init__(self, img_dim=1024, num_classes=10):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(img_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)


class LogReg(nn.Module):

    def __init__(self, img_dim=1024, num_classes=10):
        super(LogReg, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)


def train_cnn(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = nn.CrossEntropyLoss()

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_mlp(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_log_reg(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = LogReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)


def train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs):
    best_acc = best_train_acc = best_test_acc = 0.

    for _ in tqdm(range(max_epochs)):
        for _, (train_x, train_y) in enumerate(loader1):

            x = train_x.to(device).to(torch.float32) * 2. - 1.
            y = train_y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = objective(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        acc, _ = compute_acc(model, loader2, device)
        if acc > best_acc:
            best_acc = acc
            best_train_acc, _ = compute_acc(model, loader3, device)
            best_test_acc, _ = compute_acc(model, loader4, device)
        model.train()

    return best_train_acc, best_test_acc
    # return best_test_acc


def compute_acc(model, loader, device):
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float32)
            data = data * 2. - 1.
            output = model(data)
            output = nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().cpu().item()
            outputs.append(output)
    preds = torch.cat(outputs, dim=0)
    test_loss /= loader.dataset.__len__()
    acc = correct / loader.dataset.__len__()
    return acc, preds


def train_all_classifiers(train_set_loader, test_set_loader, device, batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root='data/mnist/', train=True, download=True, transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]))
    test_dataset = torchvision.datasets.MNIST(
        root='data/mnist/', train=False, download=True, transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]))

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    train_cnn_acc, test_cnn_acc = train_cnn(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    train_mlp_acc, test_mlp_acc = train_mlp(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    train_log_rec_acc, test_log_rec_acc = train_log_reg(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    return train_cnn_acc, test_cnn_acc, train_mlp_acc, test_mlp_acc, train_log_rec_acc, test_log_rec_acc


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dic, transform=None):
        super().__init__()

        self.dic = dic
        self.img = self.dic["image"]
        self.label = self.dic["class_label"]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.img[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        else:
            return image

    def __len__(self):
        return len(self.img)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dic, transform=None):
        super().__init__()

        self.dic = dic
        self.img = self.dic["image"]
        self.label = self.dic["class_label"]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.img[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        else:
            return image

    def __len__(self):
        return len(self.img)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        # description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-y", "--yaml", type=str, default=None, help="Path to config file (.yaml)")
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None, help="Path to checkpoint file (.ckpt)")
    parser.add_argument("-step", "--ddim_step", type=int, default=200, help="number of steps for ddim sampling")
    parser.add_argument("-eta", "--eta", type=float, default=1.0, help="eta for ddim sampling")
    parser.add_argument("-scale", "--scale", type=float, default=1.0, help="scale for ddim sampling")
    parser.add_argument("-bs", "--batch_size", type=int, default=500, help="Number of images to generate per batch")

    args = parser.parse_args()

    config = OmegaConf.load(args.yaml)
    model = load_model_from_config(config, args.ckpt_path)
    ddim_steps = args.ddim_step
    ddim_eta = args.eta
    scale = args.scale
    num_samples = 50000
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = args.batch_size
    n_samples_per_class = int(num_samples / len(classes))
    iter = int(n_samples_per_class/batch_size)
    sampler = DDIMSampler(model)
    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    all_samples = list()

    with torch.no_grad():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [9]).to(model.device)}
        )
    for class_label in classes:
        xc = torch.tensor(n_samples_per_class * [class_label])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

        for idx in range(iter):
            c_batch = c[idx*batch_size: (idx+1)*batch_size]

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c_batch,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            transform = transforms.Grayscale()
            x_samples_ddim = transform(x_samples_ddim)
            all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    labels = np.array(classes)
    labels = np.repeat(labels, n_samples_per_class)
    labels = torch.tensor(labels)
    train_dic = {'image': grid,
                 'class_label': labels}

    train_dataset = TrainDataset(dic=train_dic)
    train_queue = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    num_samples = 10000
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = args.batch_size
    n_samples_per_class = int(num_samples / len(classes))
    iter = int(n_samples_per_class / batch_size)
    sampler = DDIMSampler(model)
    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    all_samples = list()

    with torch.no_grad():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [9]).to(model.device)}
        )
    for class_label in classes:
        xc = torch.tensor(n_samples_per_class * [class_label])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

        for idx in range(iter):
            c_batch = c[idx * batch_size: (idx + 1) * batch_size]

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c_batch,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                         min=0.0, max=1.0)

            transform = transforms.Grayscale()
            x_samples_ddim = transform(x_samples_ddim)
            all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    labels = np.array(classes)
    labels = np.repeat(labels, n_samples_per_class)
    labels = torch.tensor(labels)
    test_dic = {'image': grid, 'class_label': labels}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    eval_dataset = TestDataset(dic=test_dic)
    eval_queue = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=128, shuffle=True)

    _, test_cnn_acc, _, test_mlp_acc, _, test_log_rec_acc = train_all_classifiers(train_queue, eval_queue, device, 128)
    print('Log reg test acc: %.4f' % (test_log_rec_acc))
    print('MLP test acc: %.4f' % (test_mlp_acc))
    print('CNN test acc: %.4f' % (test_cnn_acc))


if __name__ == "__main__":
    set_seeds(0, 0)
    main()
