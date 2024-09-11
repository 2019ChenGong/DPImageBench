import os
import numpy as np
import torch as pt
from torchvision import datasets, transforms
import matplotlib

matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def expand_vector(v, tgt_vec):
    tgt_dims = len(tgt_vec.shape)
    if tgt_dims == 2:
        return v[:, None]
    elif tgt_dims == 3:
        return v[:, None, None]
    elif tgt_dims == 4:
        return v[:, None, None, None]
    elif tgt_dims == 5:
        return v[:, None, None, None, None]
    elif tgt_dims == 6:
        return v[:, None, None, None, None, None]
    else:
        return ValueError


def flip_mnist_data(dataset):
    data = dataset.data
    flipped_data = 255 - data
    selections = np.zeros(data.shape[0], dtype=np.int)
    selections[:data.shape[0] // 2] = 1
    selections = pt.tensor(np.random.permutation(selections), dtype=pt.uint8)
    print(selections.shape, data.shape, flipped_data.shape)
    dataset.data = pt.where(selections[:, None, None], data, flipped_data)


def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path, denorm=True, save_raw=True):
    bs = mnist_mat.shape[0]
    n_to_fill = n_rows * n_cols - bs
    mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
    fill_mat = np.zeros((n_to_fill, 28, 28))
    mnist_mat = np.concatenate([mnist_mat, fill_mat])
    mnist_mat_as_list = [np.split(mnist_mat[n_rows * i:n_rows * (i + 1)], n_rows) for i in range(n_cols)]
    mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

    if denorm:
        mnist_mat_flat = denormalize(mnist_mat_flat)
    save_img(save_path + '.png', mnist_mat_flat)
    if save_raw:
        np.save(save_path + '_raw.npy', mnist_mat_flat)


def get_svhn_dataloaders(batch_size, test_batch_size, use_cuda, data_dir='data/SVHN/'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transforms_list = [transforms.ToTensor()]
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.SVHN(data_dir, split='train', download=False, transform=prep_transforms)
    tst_data = datasets.SVHN(data_dir, split='test', download=False, transform=prep_transforms)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def plot_svhn_batch(svhn_mat, n_rows, n_cols, save_path, save_raw=True):
    svhn_hw = 32
    bs = svhn_mat.shape[0]
    n_to_fill = n_rows * n_cols - bs
    svhn_mat = np.reshape(svhn_mat, (bs, 3, svhn_hw, svhn_hw))
    svhn_mat = np.transpose(svhn_mat, axes=(0, 2, 3, 1))
    fill_mat = np.zeros((n_to_fill, svhn_hw, svhn_hw, 3))
    svhn_mat = np.concatenate([svhn_mat, fill_mat])
    shvn_mat_as_list = [np.split(svhn_mat[n_rows * i:n_rows * (i + 1)], n_rows) for i in range(n_cols)]
    svhn_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in shvn_mat_as_list], axis=1)

    save_img(save_path + '.png', svhn_mat_flat)
    if save_raw:
        np.save(save_path + '_raw.npy', svhn_mat_flat)


def save_gen_labels(label_mat, n_rows, n_cols, save_path, save_raw=True):
    if save_raw:
        np.save(save_path + '_raw.npy', label_mat)
    max_labels = np.argmax(label_mat, axis=1)
    bs = label_mat.shape[0]
    n_to_fill = n_rows * n_cols - bs
    fill_mat = np.zeros((n_to_fill,), dtype=np.int) - 1
    max_labels = np.concatenate([max_labels, fill_mat])
    max_labels_flat = np.stack(np.split(max_labels, n_cols, axis=0), axis=1)
    with open(save_path + '_max_labels.txt', mode='w+') as file:
        file.write(str(max_labels_flat))


def denormalize(mnist_mat):
    mnist_mean = 0.1307
    mnist_sdev = 0.3081
    return np.clip(mnist_mat * mnist_sdev + mnist_mean, a_min=0., a_max=1.)


def save_img(save_file, img):
    plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)


def meddistance(x, subsample=None, mean_on_fail=True):
    """
  Compute the median of pairwise distances (not distance squared) of points
  in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

  Parameters
  ----------
  x : n x d numpy array
  mean_on_fail: True/False. If True, use the mean when the median distance is 0.
      This can happen especially, when the data are discrete e.g., 0/1, and
      there are more slightly more 0 than 1. In this case, the m

  Return
  ------
  median distance
  """
    if subsample is None:
        d = dist_matrix(x, x)
        itri = np.tril_indices(d.shape[0], -1)
        tri = d[itri]
        med = np.median(tri)
        if med <= 0:
            # use the mean
            return np.mean(tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = x.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(x[ind, :], None, mean_on_fail)


def dist_matrix(x, y):
    """
  Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
  """
    sx = np.sum(x ** 2, 1)
    sy = np.sum(y ** 2, 1)
    d2 = sx[:, np.newaxis] - 2.0 * x.dot(y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    d2[d2 < 0] = 0
    d = np.sqrt(d2)
    return d


def log_args(log_dir, args):
    """ print and save all args """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args_log'), 'w') as f:
        lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
        f.writelines(lines)
        for line in lines:
            print(line.rstrip())
    print('-------------------------------------------')


def parse_n_hid(n_hid, conv=False):
    """
  make sure conversion is the same everywhere
  """
    if conv:  #
        return [(s[0], [int(k) for k in s[1:].split('-')]) for s in n_hid.split(',')]
    else:  # fully connected case: just a list of linear ops
        return tuple([int(k) for k in n_hid.split(',')])


def flat_data(data, labels, device, n_labels=10, add_label=False):
    bs = data.shape[0]
    if add_label:
        gen_one_hots = pt.zeros(bs, n_labels, device=device)
        gen_one_hots.scatter_(1, labels[:, None], 1)
        labels = gen_one_hots
        return pt.cat([pt.reshape(data, (bs, -1)), labels], dim=1)
    else:
        if len(data.shape) > 2:
            return pt.reshape(data, (bs, -1))
        else:
            return data


def flatten_features(data):
    if len(data.shape) == 2:
        return data
    else:
        return pt.reshape(data, (data.shape[0], -1))


class NamedArray:
    def __init__(self, array, dim_names, idx_names):
        assert isinstance(array, np.ndarray) and isinstance(idx_names, dict)
        assert len(dim_names) == len(idx_names.keys()) and len(dim_names) == len(array.shape)
        for idx, name in enumerate(dim_names):
            assert len(idx_names[name]) == array.shape[idx] and name in idx_names
        self.array = array
        self.dim_names = dim_names  # list of dimension names in order
        self.idx_names = idx_names  # dict for the form dimension_name: [list of index names]

    def get(self, name_index_dict):
        """
    basically indexing by name for each dimension present in name_index_dict, it selects the given indices
    """
        for name in name_index_dict:
            assert name in self.dim_names
        ar = self.array
        for d_idx, dim in enumerate(self.dim_names):
            if dim in name_index_dict:
                names_to_get = name_index_dict[dim]
                # ids_to_get = [k for (k, name) in enumerate(self.idx_names[dim]) if name in names_to_get]
                ids_to_get = [self.idx_names[dim].index(name) for name in names_to_get]
                ar = np.take(ar, ids_to_get, axis=d_idx)
        return np.squeeze(ar)

    def merge(self, other, merge_dim):
        """
    merges another named array with this one:
    dimension names must be the same and in the same order
    in merge dimension: create union of index names (must be disjunct)
    in all other dimenions: create intersection of index names (must not be empty)
    """
        assert isinstance(other, NamedArray)
        assert merge_dim in self.dim_names
        assert all([n1 == n2 for n1, n2 in zip(self.dim_names, other.dim_names)])  # assert same dim names
        assert not [k for k in self.idx_names[merge_dim] if
                    k in other.idx_names[merge_dim]]  # assert merge ids disjunct
        for dim in self.dim_names:
            if dim != merge_dim:
                assert any(
                    [k for k in self.idx_names[dim] if k in other.idx_names[dim]])  # assert intersection not empty

        self_dict = {}
        other_dict = {}
        merged_idx_names = {}
        # go through dims and construct index_dict for both self and other
        for d_idx, dim in enumerate(self.dim_names):
            if dim == merge_dim:
                self_dict[dim] = self.idx_names[dim]
                other_dict[dim] = other.idx_names[dim]
                merged_idx_names[dim] = self.idx_names[dim] + other.idx_names[dim]
            else:
                intersection = [k for k in self.idx_names[dim] if k in other.idx_names[dim]]
                self_dict[dim] = intersection
                other_dict[dim] = intersection
                merged_idx_names[dim] = intersection

        # then .get both sub-arrays and concatenate them
        self_sub_array = self.get(self_dict)
        other_sub_array = other.get(other_dict)

        merged_array = np.concatenate([self_sub_array, other_sub_array], axis=self.dim_names.index(merge_dim))

        # create new NamedArray instance and return it
        return NamedArray(merged_array, self.dim_names, merged_idx_names)


def extract_numpy_data_mats():
    def prep_data(dataset):
        x, y = dataset.data.numpy(), dataset.targets.numpy()
        x = np.reshape(x, (-1, 784)) / 255
        return x, y

    x_trn, y_trn = prep_data(datasets.MNIST('data', train=True))
    x_tst, y_tst = prep_data(datasets.MNIST('data', train=False))
    np.savez('data/MNIST/numpy_dmnist.npz', x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst)

    x_trn, y_trn = prep_data(datasets.FashionMNIST('data', train=True))
    x_tst, y_tst = prep_data(datasets.FashionMNIST('data', train=False))
    np.savez('data/FashionMNIST/numpy_fmnist.npz', x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst)


def log_final_score(log_dir, final_acc):
    """ print and save all args """
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'final_score'), 'w') as f:
        lines = [f'acc: {final_acc}\n']
        f.writelines(lines)
