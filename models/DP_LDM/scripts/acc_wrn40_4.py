import numpy as np
import torch
from wideresnet import Wide_ResNet
from torch.autograd import Variable
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from torchvision import transforms
import torchvision.datasets as dset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_SDEV = [0.229, 0.224, 0.225]


def load_synth_dataset(data_file, batch_size, subset_size=None, to_tensor=False, shuffle=True,
                       source_data_scale=None, target_data_scale=None, moveaxis=False):
    if data_file.endswith('.npz'):  # allow for labels
        data_dict = np.load(data_file)
        data = data_dict['x']
        if moveaxis:
            data = np.moveaxis(data, -1, 1)
        if 'y' in data_dict.keys():
            targets = data_dict['y']
            if len(targets.shape) > 1:
                targets = np.squeeze(targets)
                assert len(targets.shape) == 1, f'need target vector. shape is {targets.shape}'
        else:
            targets = None

        if subset_size is not None:
            random_subset = np.random.permutation(data_dict['x'].shape[0])[:subset_size]
            data = data[random_subset]
            targets = targets[random_subset] if targets is not None else None

        # revert scaling if necessary
        if target_data_scale is not None:
            print(f'rescaling data of shape {data.shape} from {source_data_scale} to {target_data_scale}')
            print(f'vals as loaded: {np.min(data)}, {np.max(data)}, {data.shape}')
            assert source_data_scale is not None
            assert target_data_scale == '0_1'
            if source_data_scale == 'normed':
                mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
                sdev = np.asarray(IMAGENET_SDEV, dtype=np.float32)
            elif source_data_scale == 'bounded':
                mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
                sdev = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
            elif source_data_scale == 'normed05':
                mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
                sdev = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
            elif source_data_scale == '0_1':
                mean = np.asarray([0., 0., 0.], dtype=np.float32)
                sdev = np.asarray([1., 1., 1.], dtype=np.float32)
            else:
                raise ValueError
            data = data * sdev[None, :, None, None] + mean[None, :, None, None]
            print(f'vals as rescaled: {np.min(data)}, {np.max(data)}, {data.shape}')

        synth_data = SynthDataset(data=data, targets=targets, to_tensor=to_tensor)
    else:  # old version
        assert source_data_scale is None and target_data_scale is None
        data = np.load(data_file)
        if subset_size is not None:
            data = data[np.random.permutation(data.shape[0])[:subset_size]]
        synth_data = SynthDataset(data, targets=None, to_tensor=False)

    synth_dataloader = torch.utils.data.DataLoader(synth_data, batch_size=batch_size, shuffle=shuffle,
                                                   drop_last=False, num_workers=1)
    return synth_dataloader


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, to_tensor):
        self.labeled = targets is not None
        self.data = data
        self.targets = targets
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = torch.tensor(self.data[idx], dtype=torch.float32) if self.to_tensor else self.data[idx]
        if self.labeled:
            t = torch.tensor(self.targets[idx], dtype=torch.long) if self.to_tensor else self.targets[idx]
            return d, t
        else:
            return d


def test(model, dataloader_test, loss_fn, device):
    model.eval()
    total_correct, total_num = 0., 0.
    with torch.no_grad():
        for ims, labs in tqdm(dataloader_test):
            ims, labs = ims.to(device), labs.to(device)
            output = model(Variable(ims))
            total_correct += output.argmax(1).eq(labs).sum().cpu().item()
            total_num += ims.shape[0]
        acc = total_correct / total_num * 100
    return acc


def main():
    epochs = 10
    batch_size = 1000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading cifar10
    transformations = [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ]
    dataset_test = dset.CIFAR10(
        root="data/",
        train=False,
        transform=transforms.Compose(transformations),
        download=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Loading synthetic samples
    synth_data_file = ""  # CHANGE THIS TO POINT TO YOUR SYNTHETIC DATA (.npz)
    dataloader = load_synth_dataset(synth_data_file, batch_size, to_tensor=True, moveaxis=True)

    model = Wide_ResNet(40, 4, 0.3, 10).to(device)
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-04)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        for ims, labs in tqdm(dataloader):
            ims, labs = ims.to(device), labs.to(device)
            output = model(Variable(ims))
            loss = loss_fn(output, labs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("loss=", loss.item())

    acc = test(model, dataloader_test, loss_fn, device)
    print("Epoch: {} test accuracy={}".format(epoch, acc))


if __name__ == "__main__":
    main()
