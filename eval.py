import os
import numpy as np
import sys
import argparse

from data.dataset_loader import load_data
from utils.utils import set_logger, parse_config
from evaluation.evaluator import Evaluator


def main(config):
    set_logger(open(config.setup.workdir, 'a'))
    sensitive_train_loader, sensitive_test_loader, _ , _= load_data(config)

    syn = np.load(config.gen.log_dir)
    syn_data, syn_labels = syn["x"], syn["y"]

    evaluator = Evaluator(config)
    evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_test_loader)
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', default="PrivImage")
    parser.add_argument('--epsilon', default="1.0")
    parser.add_argument('--data_name', default="mnist_28")
    parser.add_argument('--exp_path', default="exp/privimage/mnist_28_eps1.0-2024-10-07-03-48-21")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)
    config.setup.local_rank = 0
    config.public_data.name = None
    config.setup.workdir = os.path.join(opt.exp_path, 'stdout.txt')
    config.gen.log_dir = os.path.join(opt.exp_path, 'gen', 'gen.npz')

    main(config)
