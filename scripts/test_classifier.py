import os
import sys
import argparse

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config
from evaluation.evaluator import Evaluator

def main(config):

    initialize_environment(config)

    sensitive_train_loader, sensitive_test_loader, _, _ = load_data(config)

    evaluator = Evaluator(config)
    evaluator.cal_acc_no_dp(sensitive_train_loader, sensitive_test_loader)
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="DP-MERF")
    parser.add_argument('--epsilon', '-e', default="1.0")
    parser.add_argument('--data_name', '-dn', default="mnist_28")
    parser.add_argument('--exp_description', '-ed', default="")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)
    if opt.exp_description == "":
        config.setup.workdir = "exp/{}/{}_eps{}".format(str.lower(opt.method), opt.data_name, opt.epsilon)
    else:
        config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.exp_description)

    run(main, config)
