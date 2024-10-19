import os
import numpy as np
import sys
import argparse

from data.dataset_loader import load_data
from utils.utils import set_logger, parse_config
from evaluation.evaluator import Evaluator
os.environ['MKL_NUM_THREADS'] = "1"


def main(config):

    set_logger(open(config.setup.workdir, 'a'))
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, _ , _= load_data(config)

    syn = np.load(config.gen.log_dir)
    syn_data, syn_labels = syn["x"], syn["y"]

    evaluator = Evaluator(config)
    evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader)
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', default="DP-MERF")
    parser.add_argument('--epsilon', default="10.0")
    parser.add_argument('--data_name', default="mnist_28")
    parser.add_argument('--config_suffix', '-cs', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--exp_path', default="exp/dp-merf/mnist_28_eps10.0trainval-2024-10-19-06-42-21")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)
    config.setup.local_rank = 0
    config.setup.global_rank = 0
    config.public_data.name = None

    # if not hasattr(config.setup, "workdir"):
    #     nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # if opt.resume_exp is not None:
    #     config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.resume_exp)
    # elif opt.exp_description == "":
    #     config.setup.workdir = "exp/{}/{}_eps{}-{}".format(str.lower(opt.method), opt.data_name, opt.epsilon, nowTime)
    # else:
    #     config.setup.workdir = "exp/{}/{}-{}".format(str.lower(opt.method), opt.exp_description, nowTime)

    config.setup.workdir = os.path.join(opt.exp_path, 'stdout.txt')
    config.gen.log_dir = os.path.join(opt.exp_path, 'gen', 'gen.npz')

    main(config)
