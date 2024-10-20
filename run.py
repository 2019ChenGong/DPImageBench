import os
import sys
import argparse
import datetime

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config
# from evaluation.evaluator import Evaluator
os.environ['MKL_NUM_THREADS'] = "1"

def main(config):

    initialize_environment(config)

    model = load_model(config)

    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)

    model.pretrain(public_train_loader, config.pretrain)

    model.train(sensitive_train_loader, config.train)

    syn_data, syn_labels = model.generate(config.gen)

    # evaluator = Evaluator(config)
    # evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader)
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="G-PATE")
    parser.add_argument('--epsilon', '-e', default="1.0")
    parser.add_argument('--data_name', '-dn', default="mnist_28")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)

    # if not hasattr(config.setup, "workdir"):
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if opt.resume_exp is not None:
        config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.resume_exp)
    else:
        config.setup.workdir = "exp/{}/{}_eps{}{}-{}".format(str.lower(opt.method), opt.data_name, opt.epsilon, opt.exp_description, nowTime)

    run(main, config)
